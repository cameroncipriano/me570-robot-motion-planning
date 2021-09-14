"""
Test functions for HW1
"""


import numpy as np
from numpy.lib.polynomial import poly
import me570_robot as robot


def polygon_is_visible_test():
    """
    This function should perform the following operations:
    - Create an array  test_points with dimensions [2 x 5] containing points generated uniformly at
        random using np.random.rand and scaled to approximately occupy the rectangle [0,5] [-2,2] (i.e., the
        x coordinates of the points should fall between 0 and 5, while the y coordinates between -2 and 2).
    - Obtain the polygons  polygon1 and  polygon2 from Two_Link.Polygons.
    - item:test-polygon For each polygon  polygon1,  polygon2, display a separate figure using the
        following:
        - Create the array  test_points_with_polygon by concatenating  test_points with the coordinates of
            the polygon (i.e., the coordinates of the polygon become also test points).
        - Plot the polygon (use Polygon.plot).
    - item:test-visibility For each vertex v in the polygon:
        - Compute the visibility of each point in  test_points_with_polygon with respect to that polygon
            (using Polygon.is_visible).
        - Plot lines from the vertex v to each point in  test_points_with_polygon in green if the
            corresponding point is visible, and in red otherwise.
        - Reverse the order of the vertices in the two polygons using Polygon.flip.
        - Repeat item item:test-polygon above with the reversed polygons.
    """
    test_points = np.random.rand(2, 5)

    # Scale x coordinates to uniformly cover [0, 5)
    test_points[0, :] *= 5

    # Scale y coordinates to uniformly cover [-2, 2)
    #   formula used: low + ((high - low) * random_value)
    test_points[1, :] *= 4  # high - low
    test_points[1, :] -= 2  # low

    # Obtain polygon1 and polygon2
    twoLink = robot.TwoLink()
    robot_polygons = twoLink.polygons()

    for polygon in robot_polygons:
        polygon.flip()
        polygon.plot([])


def polygon_is_collision_test():
    """
    This function is the same as polygon_is_visible_test, but use
the following:
 - Compute whether each point in  test_points_with_polygon is in collision with the polygon or not
using Polygon.is_collision.
 - Plot each point in  test_points_with_polygon in green if it is not in collision, and red
otherwise.  Moreover, increase the number of test points from 5 to 100 (i.e.,  testPoints should
have dimension [2 x 100]).
    """
    pass  # Substitute with your code


def priority_test():
    """
    The function should perform the following steps:  enumerate
 - Initialize an empty queue.
 - Add three elements (as shown in Table~tab:priority-test-inputs and in that order) to that queue.
 - Extract a minimum element.
 - Add another element (as shown in Table~tab:priority-test-inputs).
 - Check if an element is present.
 - Remove all elements by repeated extractions.  enumerate After each step, display the content of
pQueue.
    """
    pass  # Substitute with your code


polygon_is_visible_test()
