import cv2 
import os 
import glob 
from skimage import io, color, img_as_float32, img_as_ubyte
import numpy as np
from PIL import Image
import uuid

# Loads and resizes textures
def load_textures():
    # Load Textures
    img_dir = "./textures" # Directory for Textures
    data_path = os.path.join(img_dir,'*g') 
    files = glob.glob(data_path) 
    textures = []
    t_names = [] 
    for f1 in files: 
        img = io.imread(f1, as_gray = True)
         # Resizes Textures to be 200 by 200
        img = cv2.resize(img, dsize = (200,200))
        img = img_as_float32(img)
        textures.append(img)

        # Parse texture name
        texture_name = (f1.split('\\')[-1]).split('.')[0]
        t_names.append(texture_name) 

    # Returns list of textures + associated names
    return textures, t_names

# Generates and Saves Shape w/ Texture Applied
# Assumes original image is black background, white foreground
def generate_textures(img1,img_name,textures,t_names,dest_textured):
    textured_images = []

    for texture in textures:
        # Add images (background is texture, foreground is texture + shape)
        img2 = img1 + texture 
        # Compare orig texture and img2, where not equal is our shape pixels
        new_image = np.where((img2 == texture),0.5,texture)

        textured_images.append(img_as_ubyte(new_image))
    
    save_images(textured_images, img_name, t_names, dest_textured)

# Saves textured images, based off original image name and textured image names
def save_images(images, img_name, t_names, dest):    
    # Find Shape Name
    shape = img_name.split('\\')[-1].split('_')[0]

    for img,texture in zip(images,t_names):
        # Generate Name w/ UUID
        filename  = shape + "_" + texture + "_" + str(uuid.uuid1()) + ".png"
        # Save Image in Dest
        Image.fromarray(img).convert("L").save(dest + filename)