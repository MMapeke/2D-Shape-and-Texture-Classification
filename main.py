import os 
import texture_gen
import glob 

from generate_shapes import ShapeGenerator
from skimage import io, color, img_as_float32

def main():
    # Generate 2d Shape Images
    # DATASETS: 3 Shapes, 5 Shapes, 5 Shapes w/ Random Size

    num_images = 350 # Number of images per each shape
    # Destination path for original 2D images
    destination_orig = "./datasets/three_shapes/original/" 
    print("Generating Images and saving to " + destination_orig)

    generator = ShapeGenerator(destination_orig, num_images)
    generator.generate_shapes()
    
    # Apply Textures to Images
    # NOTE: destination_textured should end in '/'
    destination_textured = "./datasets/three_shapes/textured/" 
    print("Generating Texutes and saving to " + destination_textured)
    textures, t_names = texture_gen.load_textures()

    # Load original image shapes + generate new images to save
    data_path = os.path.join(destination_orig,'*g') 
    files = glob.glob(data_path) 
    for f1 in files: 
        img = io.imread(f1, as_gray = True)
        # print(img.shape)
        img = img_as_float32(img)
        
        texture_gen.generate_textures(img, f1, textures, t_names, destination_textured)

    # TODO: Learning and Classification
     

# TODO: next commit datasets in zip + all the code
if __name__ == '__main__':
    main()