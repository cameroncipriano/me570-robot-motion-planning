"""
Test functions for HW1
"""

import numpy as np
import matplotlib.pyplot as plt
import me570_robot as robot
import me570_queue as PriorityQueue


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
    two_link = robot.TwoLink()
    robot_polygons = two_link.polygons()

    for polygon in robot_polygons:
        test_points_with_polygon = np.hstack((polygon.vertices, test_points))
        plt.figure()
        polygon.plot([])
        for i in range(polygon.vertices.shape[1]):
            vertex = np.vstack(polygon.vertices[:, i])
            for j in range(test_points_with_polygon.shape[1]):
                point_to_test = np.vstack(test_points_with_polygon[:, j])
                visible = polygon.is_visible(i, point_to_test)

                values = np.hstack((vertex, point_to_test))
                x_vals = values[0, :]
                y_vals = values[1, :]

                if visible[0]:
                    plt.plot(x_vals,
                             y_vals,
                             'g',
                             marker='o',
                             markeredgecolor='k',
                             markerfacecolor='k',
                             linewidth=1,
                             alpha=0.8)
                else:
                    plt.plot(x_vals,
                             y_vals,
                             'r',
                             marker='o',
                             markeredgecolor='k',
                             markerfacecolor='k',
                             linewidth=0.5,
                             alpha=0.8)
        plt.show()

        polygon.flip()
        plt.figure()
        polygon.plot([])

        for i in range(polygon.vertices.shape[1]):
            vertex = np.vstack(polygon.vertices[:, i])
            for j in range(test_points_with_polygon.shape[1]):
                point_to_test = np.vstack(test_points_with_polygon[:, j])
                visible = polygon.is_visible(i, point_to_test)

                values = np.hstack((vertex, point_to_test))
                x_vals = values[0, :]
                y_vals = values[1, :]

                if visible[0]:
                    plt.plot(x_vals,
                             y_vals,
                             'g',
                             marker='o',
                             markeredgecolor='k',
                             markerfacecolor='k',
                             linewidth=1,
                             alpha=0.8)
                else:
                    plt.plot(x_vals,
                             y_vals,
                             'r',
                             marker='o',
                             markeredgecolor='k',
                             markerfacecolor='k',
                             linewidth=0.5,
                             alpha=0.8)
        plt.show()


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

    test_points = np.random.rand(2, 100)

    # Scale x coordinates to uniformly cover [0, 5)
    test_points[0, :] *= 5

    # Scale y coordinates to uniformly cover [-2, 2)
    #   formula used: low + ((high - low) * random_value)
    test_points[1, :] *= 4  # high - low
    test_points[1, :] -= 2  # low

    # Obtain polygon1 and polygon2
    two_link = robot.TwoLink()
    robot_polygons = two_link.polygons()

    for polygon in robot_polygons:
        test_points_with_polygon = np.hstack((polygon.vertices, test_points))
        plt.figure()
        polygon.plot([])

        green_x = []
        green_y = []
        red_x = []
        red_y = []

        flagged_points = polygon.is_collision(test_points_with_polygon)
        for i, point in enumerate(test_points_with_polygon.T):
            x_point = point[0]
            y_point = point[1]
            if flagged_points[i] is True:
                red_x.append(x_point)
                red_y.append(y_point)
            else:
                green_x.append(x_point)
                green_y.append(y_point)

        plt.scatter(green_x, green_y, color='green')
        plt.scatter(red_x, red_y, color='red')

        plt.show()

        green_x.clear()
        green_y.clear()
        red_x.clear()
        red_y.clear()

        polygon.flip()
        plt.figure()
        polygon.plot([])

        flagged_points = polygon.is_collision(test_points_with_polygon)
        for i, point in enumerate(test_points_with_polygon.T):
            x_point = point[0]
            y_point = point[1]
            if flagged_points[i] is True:
                red_x.append(x_point)
                red_y.append(y_point)
            else:
                green_x.append(x_point)
                green_y.append(y_point)

        plt.scatter(green_x, green_y, color='green')
        plt.scatter(red_x, red_y, color='red')

        plt.show()


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
    my_queue = PriorityQueue.Priority()

    my_queue.insert("Oranges", 4.5)
    my_queue.insert("Apples", 1)
    my_queue.insert("Bananas", 2.7)
    print(*my_queue.queue, sep=", ")

    key, cost = my_queue.min_extract()
    print(f"({key}, {cost})")

    my_queue.insert("Cantaloupe", 3)
    print(*my_queue.queue, sep=", ")

    print(f"Oranges is in my queue? --> {my_queue.is_member('Oranges')}")
    print(f"Milk is in my queue? --> {my_queue.is_member('Milk')}")

    while True:
        (key, value) = my_queue.min_extract()
        if (key is None and value is None):
            break

        print(f"Removed: ({key}, {value}) --> remaining: ", end=" ")
        print(*my_queue.queue, sep=", ")


priority_test()