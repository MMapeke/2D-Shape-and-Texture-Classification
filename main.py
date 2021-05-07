import numpy as np 
from skimage import io, color, img_as_float32
import matplotlib.pyplot as plt
import texture_gen

def main():
    # ASSUMING GRAYSCALE FOR EVERYTHING RN

    # Load and display Test File
    img = io.imread("sample-shape.png", as_gray = True)
    img = img_as_float32(img)
    plt.imshow(img, cmap="gray")
    plt.show()
    
    # Loads and displays textured shapes
    new_images = texture_gen.generate_textures(img)
    for texture in new_images:
        plt.imshow(texture, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()