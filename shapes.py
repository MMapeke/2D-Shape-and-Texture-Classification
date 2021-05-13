from abc import ABC, abstractmethod

import numpy as np
import io
from PIL import Image
import uuid
from os import path

# 2D shape generator, each shape is saved in png file
class AbstractShape(ABC):
    def __init__(self, destination, painter):
        self.painter = painter
        self.destination = destination
        self.radius = None
        self.x = None
        self.y = None

    # Sets background canvas color
    def set_bg_color(self):
        color = [0,0,0] # All Black

        self.painter.fillcolor(color[0], color[1], color[2])
        self.painter.color(color[0], color[1], color[2])
        self.painter.penup()
        self.painter.setposition(-160, 160)
        self.painter.pendown()
        self.painter.begin_fill()

        self.painter.goto(160, 160)
        self.painter.goto(160, -160)
        self.painter.goto(-160, -160)
        self.painter.goto(-160, 160)

        self.painter.end_fill()
        self.painter.penup()

    # Set shape fill and draw color
    def set_shape_color(self):
        color = [255,255,255] # all white
        self.painter.fillcolor(color[0], color[1], color[2])
        self.painter.color(color[0], color[1], color[2])
        self.painter.penup()

    # Sets and randomizes parameters for each shape to generate
    def set_params(self):
        self.painter.reset()

        self.set_bg_color()
        self.set_shape_color()

        # Set Shape Size/Perimeter
        self.radius = np.random.randint(20, 75)
        # self.radius = 40

        # Set Shape Rotation
        self.rotation = np.deg2rad(np.random.randint(-180, 180))
        # self.rotation = 0

        # Set Center Coordinates
        self.x = np.random.randint(-80 + self.radius, 80 - self.radius)
        self.y = np.random.randint(-80 + self.radius, 80 - self.radius)
        # self.x = self.y = 0

    # Saves Drawing with unique name as a png file
    # The name of the save image is as follows : [Type of shape]_[UUID].png
    # NOTE: Installed GhostScript on Windows for this to work
    def save_drawing(self):
        ps = self.painter.getscreen().getcanvas().postscript(
            colormode='color', pageheight=199, pagewidth=199
        )
        im = Image.open(io.BytesIO(ps.encode('utf-8')))
        im.save(path.join(
            self.destination,
            self.__class__.__name__ + "_" + str(uuid.uuid1()) + '.png'
        ), quality=100, format='png')

    # Draws shape relative to parameters
    def draw(self):
        self.painter.penup()
        shape_coordinates = self.get_shape_coordinates()
        r_coordinates = []

        # Calculates rotated shape coordinates
        for item in shape_coordinates:
            r_coordinates.append(
                (
                    (item[0] - self.x) * np.cos(self.rotation) -
                    (item[1] - self.y) * np.sin(self.rotation) + self.x,

                    (item[0] - self.x) * np.sin(self.rotation) +
                    (item[1] - self.y) * np.cos(self.rotation) + self.y
                )
            )

        self.painter.goto(r_coordinates[-1])

        self.painter.pendown()
        self.painter.begin_fill()

        # for idx, coord in enumerate(r_coordinates):
        for coord in r_coordinates:
            self.painter.goto(coord)
            # if self.should_break and self.should_break == idx:
            #     self.painter.end_fill()
            #     self.painter.begin_fill()

        self.painter.end_fill()
        self.painter.hideturtle()

    def generate(self):
        """
            Generate an image that contains a shape drown inside it, in function
            of the set of random parameters that where configured in the
            function ‘__set_random_params‘.

        :return: None
        """
        self.set_params()
        self.draw()
        self.save_drawing()

    @abstractmethod
    def get_shape_coordinates(self):
        # Calculates coordinates of points with no rotatation relative to center
        raise NotImplementedError()

# 2D shape generator for polygon with arbritary number of sides/vertices
class AbstractPolygonShape(AbstractShape, ABC):

    number_of_vertices = None
    # should_break = None

    def get_shape_coordinates(self):

        if not self.number_of_vertices:
            raise NotImplementedError(
                "The number of vertices must be specified in sub classes."
            )

        coordinates = []
        for vertex in range(self.number_of_vertices):
            coordinates.append(
                (
                    self.radius * np.cos(
                        2 * np.pi * vertex / self.number_of_vertices
                    ) + self.x,
                    self.radius * np.sin(
                        2 * np.pi * vertex / self.number_of_vertices
                    ) + self.y
                )
            )
        return coordinates


class Triangle(AbstractPolygonShape):

    number_of_vertices = 3

class Square(AbstractPolygonShape):

    number_of_vertices = 4

class Pentagon(AbstractPolygonShape):

    number_of_vertices = 5

class Hexagon(AbstractPolygonShape):

    number_of_vertices = 6

class Heptagon(AbstractPolygonShape):

    number_of_vertices = 7

class Circle(AbstractShape):

    def draw(self):

        self.painter.setposition(self.x, self.y - self.radius)

        self.painter.pendown()
        self.painter.begin_fill()
        self.painter.ht()
        self.painter.circle(self.radius)
        self.painter.end_fill()

    def get_shape_coordinates(self):
        pass