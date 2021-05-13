import turtle 
import shapes
from tqdm import tqdm

class ShapeGenerator:

    # Shapes we will be generating
    GENERATOR_CLASSES = [
        shapes.Circle,
        shapes.Triangle,
        shapes.Square,
        shapes.Pentagon,
        shapes.Hexagon,
        shapes.Heptagon
    ]

    def __init__(self,destination,num_images):
        turtle.colormode(255) # Use ints or float range for colors

        turtle.setup(width=200, height=200)
        turtle.hideturtle()
        turtle.tracer(False) # Hide Animations for speed-up

        self.painter = turtle.Turtle()

        self.num_images = num_images

        self.shapes = []
        for generator in self.GENERATOR_CLASSES:
            self.shapes.append(generator(destination,self.painter)) 

    def generate_shapes(self):
        for _ in tqdm (range (self.num_images), desc="Loading..."):
            for shape in self.shapes:
                shape.generate()
