import texture_gen
from generate_shapes import ShapeGenerator

def main():
    # Generate 2d Shape Images
    # DATASETS: 3 Shapes, 5 Shapes, 5 Shapes w/ Random Size

    num_images = 2 # Number of images per each shape
    # Destination path for original 2D images
    destination_orig = "./datasets/three_shapes/original" 
    print("Generating Images and saving to " + destination_orig)

    generator = ShapeGenerator(destination, num_images)
    generator.generate_shapes()
    
    # Apply Textures to Images
    destination_textured = "./datasets/three_shapes/textured" 
    textures, t_names = texture_gen.load_textures()

    # Load original image shapes + generate new images to save
    data_path = os.path.join(destination_orig,'*g') 
    files = glob.glob(data_path) 
    for f1 in files: 
        img = io.imread(f1, as_gray = True)
        # print(img.shape)
        img = img_as_float32(img)
        
        texture_gen.generate_textures(img, textures)

    # TODO: Learning and Classification
     

if __name__ == '__main__':
    main()