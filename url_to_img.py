import requests
from PIL import Image
from io import BytesIO

def save_image_from_url(url, save_path):
    try:
        # Send a GET request to the URL to fetch the image data
        response = requests.get(url)
        if response.status_code == 200:
            # Open the image data using BytesIO
            image_data = BytesIO(response.content)
            # Open the image using PIL
            img = Image.open(image_data)
            # Save the image to the specified path
            img.save(save_path)
            print(f"Image saved successfully to: {save_path}")
        else:
            print(f"Failed to fetch image from URL: {url}")
    except Exception as e:
        print(f"Error saving image: {e}")

# URL of the image
image_url = "https://lp2.hm.com/hmgoepprod?set=quality%5B79%5D%2Csource%5B%2F0a%2F64%2F0a646eb5ea9d4b598ab17cde5621827f51e0ba6f.jpg%5D%2Corigin%5Bdam%5D%2Ccategory%5Bkids_baby_boy_clothing_tshirtsshirts_tshirts%5D%2Ctype%5BDESCRIPTIVESTILLLIFE%5D%2Cres%5Bm%5D%2Chmver%5B2%5D&call=url[file:/product/main]"
# Local path where you want to save the image
save_path = "saved_image.jpg"

# Call the function to save the image
save_image_from_url(image_url, save_path)
