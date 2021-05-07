import cv2 
import os 
import glob 
from skimage import io, color, img_as_float32
import numpy as np

# Loads and resizes textures
def load_textures():
    # Load Textures
    img_dir = "./textures" # Directory for Textures
    data_path = os.path.join(img_dir,'*g') 
    files = glob.glob(data_path) 
    textures = [] 
    for f1 in files: 
        img = io.imread(f1, as_gray = True)
         # Resizes Textures to be 200 by 200
        img = cv2.resize(img, dsize = (200,200))
        img = img_as_float32(img)
        textures.append(img) 

    # Returns list of textures
    return textures
    # TODO: Return second array of texture names, so making files simple later

# Generates and Returns Shapes w/ Texture (as np arrays)
def generate_textures(img1):
    textures = load_textures()
    textured_images = []

    for texture in textures:
        # Add images (background is texture, foreground is texture + shape)
        img2 = img1 + texture 
        # Compare orig texture and img2, where not equal is our shape pixels
        new_image = np.where((img2 == texture),0.5,texture)
        textured_images.append(new_image)
    
    return textured_images
