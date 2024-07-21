import bpy
import math
import os
import pickle
import numpy as np

def render_image(obj_file_path,output_image_path,x,y,z,a,b,c):
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

    # Position the camera for a front-right view
    camera.location = (x, y, z)
    camera.rotation_euler = (a,b,c) 
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


if __name__ == "__main__":
    dataset_path = "C:/Users/tejor/OneDrive/Desktop/SVTON/1937/model_0.8.obj"  
    # x,y,z= -3.4,-4.2,4 # left
    # a,b,c= math.radians(60), 0, math.radians(-45) # left
    # x,y,z= 4.2, -4.2, 4 #right
    # a,b,c= math.radians(60), 0, math.radians(45) #right
    x,y,z= 0.45, -8, 0.9 #center
    a,b,c= math.pi / 2, 0, 0 #center
    output_path = f"C:/Users/tejor/OneDrive/Desktop/SVTON/trial/{x}_{y}_{z}.png"
    render_image(dataset_path,output_path,x,y,z,a,b,c)
