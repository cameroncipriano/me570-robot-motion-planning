"""
Representation of a simple robot used in the assignments
"""

import numpy as np
import me570_geometry as gm


class TwoLink:
    """ A class containing methods for a two-link manipulator. """
    def __init__(self):
        add_y_reflection = lambda vertices: np.hstack(
            [vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])

        vertices1 = np.array([[0, 5], [-1.11, -0.511]])
        vertices1 = add_y_reflection(vertices1)
        vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                              [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
        vertices2 = add_y_reflection(vertices2)
        self.polygons = (gm.Polygon(vertices1), gm.Polygon(vertices2))
