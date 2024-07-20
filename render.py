import bpy
import math
import os
import pickle

def render_image(obj_file_path,output_image_path):
    # Clear existing meshes, lights, and cameras
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge()

    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=obj_file_path)
    # Set up the camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    camera.location = (0.45, -8, 0.9)
    camera.rotation_euler = (math.pi / 2, 0, 0) 
    camera.data.lens = 70  


    # Set up the light
    light_data = bpy.data.lights.new(name='Light', type='POINT')
    light = bpy.data.objects.new('Light', light_data)
    bpy.context.collection.objects.link(light)

    light.location = camera.location

    # Set the render resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # Set the output file path
    bpy.context.scene.render.filepath = output_image_path
    # Render the scene
    bpy.ops.render.render(write_still=True)
    print("Rendering completed and saved to", output_image_path)

def read_mtl_file(filepath):
    materials = {}
    current_material = None

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue  # Skip empty lines
            if parts[0] == 'newmtl':
                current_material = parts[1]
                materials[current_material] = {}
            elif current_material is not None:
                key = parts[0]
                values = parts[1:]
                materials[current_material][key] = values
    
    return materials
def edit_material(materials, material_name, property_name, new_values):
    if material_name in materials:
        materials[material_name][property_name] = new_values
    else:
        print(f"Material {material_name} not found")
def write_mtl_file(filepath, materials):
    with open(filepath, 'w') as file:
        for material, properties in materials.items():
            file.write(f'newmtl {material}\n')
            for key, values in properties.items():
                file.write(f'{key} {" ".join(values)}\n')
            file.write('\n')

def read_write_img(obj_folder,output_folder,sizes):
    obj_file = "model_0.8.obj"
    mtl_file = "model_0.8.mtl"
    output_color_image = f"object_image_{sizes[0]}_{sizes[1]}_{sizes[2]}.png"
    output_segment_image = f"segment_image_{sizes[0]}_{sizes[1]}_{sizes[2]}.png"
    obj_file_path = os.path.join(obj_folder, obj_file)
    mtl_file_path = os.path.join(obj_folder, mtl_file)
    color_output_path = os.path.join(output_folder, output_color_image)
    segment_output_path = os.path.join(output_folder, output_segment_image)

    render_image(obj_file_path,color_output_path)
    materials = read_mtl_file(mtl_file_path)
    edit_material(materials, 'model_0.8', 'map_Kd', ['seg_0.8_new.png'])
    write_mtl_file(mtl_file_path, materials)
    render_image(obj_file_path,segment_output_path)

def size_extractor(pikle_file_path):
    with open(pikle_file_path, 'rb') as f:
        data = pickle.load(f)
    sizes = {}
    for key, value in data.items():
        parts = value.split('_')
        if len(parts) == 4:
            garment_type = parts[0]
            upper_body_size = parts[1]
            lower_body_size = parts[2]
            sizes[key] = (garment_type,upper_body_size, lower_body_size)
    return sizes

def main(dataset_path,output_path,pikle_file_path):
    sizes = size_extractor(pikle_file_path)
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for subfolder in os.listdir(folder_path):
                try:
                    obj_folder = os.path.join(folder_path, subfolder)
                    output_folder = os.path.join(output_path, folder) 
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    key = folder+"/"+subfolder
                    read_write_img(obj_folder,output_folder,sizes[key])
                except:
                    continue

if __name__ == "__main__":
    dataset_path = "C:/Users/tejor/OneDrive/Desktop/SVTON/sizer_dataset/scans/dataset"
    output_path = "C:/Users/tejor/OneDrive/Desktop/SVTON/processed_dataset_new"
    pikle_file_path= 'C:/Users/tejor/OneDrive/Desktop/SVTON/sizer_dataset/scans/sizing_data.pkl'
    main(dataset_path,output_path,pikle_file_path)








