import turtle 
import shapes

class ShapeGenerator:

    # Shapes we will be generating
    GENERATOR_CLASSES = [
        shapes.Circle,
        shapes.Triangle,
        shapes.Square
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
        for _ in range(self.num_images):
            for shape in self.shapes:
                shape.generate()