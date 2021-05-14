## 2D-Shape-and-Texture-Classification

https://docs.google.com/presentation/d/1Bj9CBX28tg_7ktYXXeGqa1CvohkoYT_kKzeUSX37KsA/edit?usp=sharing

# Generating Dataset

In main.py:
-num_images is the number of images to generate for each shape, and also the number of images that will be in each class. 
-destination_orig is the folder to store the original shape images
-destination_textured is the folder to store the final shape + texture images

In shapes.py:
-This is where one can add new shapes. To add polygons, follow the design for any oteher polygon and just change the class name and number of sides. 

In generate_shapes.py:
-GENERATOR_CLASSES is a list of the Shape classes we will generate and this can be adjusted to add or remove shapes

In texture_gen.py:
-img_dir is the directory for the texture file images

In order to generate a dataset run 'python main.py'.

# Running Model

To run any model, just call 'python model_name.py' and in each of these files directory is the folder that holds the dataset with the textured images in their respective folders.

model.py is my custom model