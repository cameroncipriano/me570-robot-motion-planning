import matplotlib.pyplot as plt
import numpy as np
import me570_geometry as gm
import me570_potential as pot


def sphere_testCollision():
    """
    Generates one figure with a sphere (with arbitrary parameters) and  nb_points=100 random points that
are colored according to the sign of their distance from the sphere (red for negative, green for
positive). Generates a second figure in the same way (and the same set of points) but flipping the
sign of the radius  r of the sphere. For each sampled point, plot also the result of the output
pointsSphere.
    """
    plt.figure()

    center = np.vstack((0, 0))
    radius = 3
    influence_dist = 1
    sphere = gm.Sphere(center, radius, influence_dist)

    test_points = np.random.uniform(-radius * 2, radius * 2, size=(2, 100))

    collision_x = []
    collision_y = []
    non_collision_x = []
    non_collision_y = []

    flag_points = sphere.is_collision(test_points)
    for i, point in enumerate(test_points.T):
        x_point = point[0]
        y_point = point[1]
        if flag_points[i] == True:
            collision_x.append(x_point)
            collision_y.append(y_point)
        else:
            non_collision_x.append(x_point)
            non_collision_y.append(y_point)

    sphere.plot('k')
    plt.scatter(collision_x, collision_y, color='r', zorder=10)
    plt.scatter(non_collision_x, non_collision_y, color='g', zorder=10)

    plt.show()

    plt.figure()
    collision_x.clear()
    collision_y.clear()
    non_collision_x.clear()
    non_collision_y.clear()
    flag_points.clear()

    sphere.flip()

    flag_points = sphere.is_collision(test_points)
    for i, point in enumerate(test_points.T):
        x_point = point[0]
        y_point = point[1]
        if flag_points[i] == True:
            collision_x.append(x_point)
            collision_y.append(y_point)
        else:
            non_collision_x.append(x_point)
            non_collision_y.append(y_point)

    sphere.plot('k')
    plt.scatter(collision_x, collision_y, color='r', zorder=10)
    plt.scatter(non_collision_x, non_collision_y, color='g', zorder=10)

    plt.show()


# Running me570_potential file
