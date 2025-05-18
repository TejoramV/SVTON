# stage1_sizer.py  (update only the dataset + main paths section)

import os, re, glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

SAVE_DEBUG = False                                   # flip to False later
DEBUG_DIR  = "C:/Users/Tejoram/Desktop/SVTON/Debug"                        
os.makedirs(DEBUG_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Dataset that walks sub-directories of processed_dataset_new
# ------------------------------------------------------------
class VTONDataset(Dataset):
    """
    Each sample returns:
      person_img : RGB (garment region zeroed)
      garment_img: RGB (isolated upper-garment patch)
      size_code  : one-hot tensor (S=0, M=1, L=2)
      gt_mask    : binary segmentation mask for the upper garment
      size_label : integer class 0/1/2
    """
    def __init__(self, root_dir, size_map, split='upper', im_res=256):
        """
        root_dir :  processed_dataset_new/
        split    :  'upper' or 'lower' (which garment to model)
        """
        self.files = []            # full path to object_image_*.png
        for obj_path in glob.iglob(os.path.join(root_dir, '*', 'object_image_*.png')):
            self.files.append(obj_path)
        self.size_map = size_map
        self.split    = split      # decide which mask colour channel to use
        # self.T = transforms.Compose([
        #     transforms.Resize((im_res, im_res)),            
        #     transforms.ToTensor()

        # ])
        # 1️⃣  Put Resize BEFORE ToTensor  (works for images too)
        self.T_img  = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        # 2️⃣  Separate transform for masks — use NEAREST and keep them single‑channel
        self.T_mask = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()                       # returns (1, 256, 256)
        ])

    def __len__(self): return len(self.files)

    def _parse_sizes(self, fname):
        m = re.search(r'_(s|m|l|xl)_(s|m|l|xl)\.png$', fname)
        return m.group(1), m.group(2)          # upper_size, lower_size


    def _extract_mask(self, seg_np):
        """
        Return a clean binary garment mask (float32 0/1) with:
        • colour-threshold
        • height gate   (keep rows y < 850)
        • 3×3 opening   (remove 1-px specks)
        • largest-blob  selection (keeps only the garment)
        """
        H, W, _ = seg_np.shape
        row_mask = (np.arange(H)[:, None] < 850)          # (H,1)

        # 1) raw binary
        if self.split == 'upper':                         # pure blue ≈ red<20 & green<20
            raw = ((seg_np[:, :, 0] < 20) &
                (seg_np[:, :, 1] < 20) & row_mask)
        else:                                            # lower → strong green, low R & B
            raw = ((seg_np[:, :, 0] < 20) &
                (seg_np[:, :, 2] < 20) &
                (seg_np[:, :, 1] > 128) & row_mask)

        # 2) morph opening (3×3) to kill tiny dots
        raw_u8 = raw.astype(np.uint8) * 255
        opened = cv2.morphologyEx(
            raw_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        # 3) keep only the largest connected component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # 0 = background
            cleaned = (labels == largest).astype(np.float32)
        else:
            cleaned = (opened > 0).astype(np.float32)

        return cleaned

    def __getitem__(self, idx):
        obj_path = self.files[idx]
        seg_path = obj_path.replace('object_image', 'segment_image')
        obj_np = np.array(Image.open(obj_path).convert('RGB')) / 255.
        seg_np = np.array(Image.open(seg_path).convert('RGB'))

        upper_size, lower_size = self._parse_sizes(os.path.basename(obj_path))
        size_str  = upper_size if self.split == 'upper' else lower_size
        size_lbl  = self.size_map[size_str]

        mask_np   = self._extract_mask(seg_np)                 # H×W
        person_np = obj_np * (1 - mask_np[...,None])           # zero garment region
        garment_np= obj_np * (mask_np[...,None])               # only garment

        # to tensor & resize
        # person   = self.T(person_np).float() 
        # garment  = self.T(garment_np).float() 
        # mask     = self.T(mask_np[None,:,:]).float()           # 1×H×W

        # ----- inside __getitem__ ------------------------------------
        person_img  = Image.fromarray((person_np * 255).astype(np.uint8))   # H×W×3
        garment_img = Image.fromarray((garment_np * 255).astype(np.uint8))
        mask_img    = Image.fromarray((mask_np * 255).astype(np.uint8))     # H×W (grayscale)

        if SAVE_DEBUG:
            # build a unique stem from the original filename
            stem = os.path.basename(obj_path).replace("object_image_", "").rsplit(".",1)[0]

            # ./debug_samples/person_00001_s_m.png  etc.
            person_img.save (os.path.join(DEBUG_DIR, f"person_{stem}.png"))
            garment_img.save(os.path.join(DEBUG_DIR, f"garment_{stem}.png"))
            mask_img.save   (os.path.join(DEBUG_DIR, f"mask_{stem}.png"))        

        person  = self.T_img(person_img).float()      # (3, im_res, im_res)
        garment = self.T_img(garment_img).float()
        mask    = self.T_mask(mask_img).float()       # (1, im_res, im_res)


        size_code           = torch.zeros(len(self.size_map))
        size_code[size_lbl] = 1.

        return person, garment, size_code, mask, size_lbl


# ------------------------------------------------------------
# 2.  (network / losses identical to previous answer)
#     paste Stage1UNet, dice_loss, train_size_classifier,
#     train_mask_generator from the earlier message here.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.  Train -- point to your processed_dataset_new root
# ------------------------------------------------------------

# 2) U-Net + Size Classifier
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Stage1UNet(nn.Module):
    def __init__(self, num_sizes):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(7, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(2)
        # size-classifier head on bottleneck
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_sizes)
        )
        # decoder
        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = ConvBlock(512+512, 256)
        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec4 = ConvBlock(256+256, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = ConvBlock(128+128, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = ConvBlock(64+64, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, person, garment, size_code):
        #concat (B,7,H,W)
        # x = torch.cat([
        #     person,
        #     garment,
        #     size_code[:,None,None,:].expand(-1,1,person.size(2),person.size(3))
        # ], dim=1)

        B, _, H, W = person.shape
        size_plane = size_code.argmax(1, keepdim=True).float()      # (B,1)
        size_plane = size_plane[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([person, garment, size_plane], dim=1)         # (B,7,H,W)

        # B, S = size_code.shape          # S = 4
        # size_map = size_code.view(B, S, 1, 1).expand(-1, S, person.size(2), person.size(3))
        # x = torch.cat([person, garment, size_map], dim=1)   # (B,10,H,W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        logits = self.classifier(e5)  # size classification
        # decoder with skips
        d5 = self.dec5(torch.cat([self.up5(e5), e4],1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3],1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2],1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1],1))
        mask = torch.sigmoid(self.final(d2))
        return mask, logits

# 3) Losses
def dice_loss(pred, target, eps=1e-6):
    num = 2*(pred*target).sum(dim=(1,2,3)) + eps
    den = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - (num/den).mean()

# 4) Training functions
def train_size_classifier(model, loader, device, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce  = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for person, garment, size_code, _, label in tqdm(loader, desc=f"CLS Ep{ep}"):
            person, garment, label = person.to(device), garment.to(device), label.to(device)
            size_code = size_code.to(device)
            opt.zero_grad()
            _, logits = model(person, garment, size_code)
            loss = ce(logits, label)
            loss.backward()
            opt.step()

def train_mask_generator(model, loader, device, epochs=20, λ_dice=1.0, λ_size=0.1):
    # freeze encoder & classifier
    for name, p in model.named_parameters():
        if name.startswith('enc') or name.startswith('classifier'):
            p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for person, garment, size_code, mask, label in tqdm(loader, desc=f"SEG Ep{ep}"):
            person, garment, size_code, mask, label = \
                person.to(device), garment.to(device), size_code.to(device), mask.to(device), label.to(device)
            opt.zero_grad()
            pred, logits = model(person, garment, size_code)
            loss = (bce(pred, mask) +
                    λ_dice*dice_loss(pred, mask) +
                    λ_size*ce(logits, label))
            loss.backward()
            opt.step()
if __name__ == "__main__":
    root  = "C:/Users/Tejoram/Desktop/SVTON/processed_dataset_new"   # ← adjust!
    device= torch.device("cuda")
    size_map = {'s':0, 'm':1, 'l':2, 'xl':3}

    # choose which garment to learn  ('upper' or 'lower')
    split_type = 'upper'                     

    ds = VTONDataset(root, size_map, split=split_type, im_res=256)
    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    num_workers=4, pin_memory=True)

    model = Stage1UNet(num_sizes=len(size_map)).to(device)

    # A) Pre-train encoder on size-classification
    train_size_classifier(model, dl, device, epochs=2)

    # B) Fine-tune decoder for mask generation
    train_mask_generator(model, dl, device, epochs=1)

    torch.save(model.state_dict(),
               f"C:/Users/Tejoram/Desktop/SVTON/output/stage1_sizer_unet_{split_type}.pth")
