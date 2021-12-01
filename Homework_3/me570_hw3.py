"""
Defines a module to test general methods in other files
"""
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
        if flag_points[i]:
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
        if flag_points[i]:
            collision_x.append(x_point)
            collision_y.append(y_point)
        else:
            non_collision_x.append(x_point)
            non_collision_y.append(y_point)

    sphere.plot('k')
    plt.scatter(collision_x, collision_y, color='r', zorder=10)
    plt.scatter(non_collision_x, non_collision_y, color='g', zorder=10)

    plt.show()


# sphere_testCollision()

attr_pot = pot.Attractive({'shape': 'quadratic', 'x_goal': np.vstack((0, 0))})

f_handle_attr = lambda point: attr_pot.eval(point)
plt.figure()
gm.field_plot_threshold(f_handle_attr)

central_path = gm.Sphere(np.vstack((0, 0)), -6, 5)
central_sphere = pot.RepulsiveSphere(central_path)

plt.show()

f_handle_repulsive = lambda point: central_sphere.eval(point)
gm.field_plot_threshold(f_handle_repulsive, 6, 200)
plt.show()

# world = pot.SphereWorld()
f_handle_total = lambda point: world_potential.eval(point)

# potential_1 = {
#     'shape': 'quadratic',
#     'x_goal': np.vstack(world.x_goal[:, 0]),
#     'repulsive_weight': 0.01
# }
# world_potential = pot.Total(world, potential_1)

# plt.figure()
# gm.field_plot_threshold(f_handle_total, 10, 200)
# plt.show()

# potential_2 = {
#     'shape': 'quadratic',
#     'x_goal': np.vstack(world.x_goal[:, 1]),
#     'repulsive_weight': 0.01
# }

# world_potential = pot.Total(world, potential_2)

# plt.figure()
# gm.field_plot_threshold(f_handle_total, 10, 200)
# plt.show()

# potential_3 = {
#     'shape': 'conic',
#     'x_goal': np.vstack(world.x_goal[:, 0]),
#     'repulsive_weight': 0.01
# }

# world_potential = pot.Total(world, potential_3)

# plt.figure()
# gm.field_plot_threshold(f_handle_total, 10, 200)
# plt.show()

# potential_4 = {
#     'shape': 'conic',
#     'x_goal': np.vstack(world.x_goal[:, 1]),
#     'repulsive_weight': 0.01
# }

# world_potential = pot.Total(world, potential_4)

# plt.figure()
# gm.field_plot_threshold(f_handle_total, 10, 200)
# plt.show()

# my_potential_planner = pot.Planner()
# my_potential_planner.run_plot()

# world = pot.SphereWorld()
# goal_loc = world.x_goal
# potential = {
#     'x_goal': np.vstack(world.x_goal[:, 1]),
#     'shape': 'quadratic',
#     'repulsive_weight': 35
# }
# f_handle = lambda point: pot.clfcbf_control(point, world, potential)
# gm.field_plot_threshold(f_handle, 10, 20)
# world.plot()
# plt.show()

# total_potential = pot.Total(world, potential)
# total_pot_handle = lambda point: total_potential.eval(point)
# total_pot_grad_handle = lambda point: -total_potential.grad(point)

# plt.figure()
# gm.field_plot_threshold(total_pot_handle, 10, 200)

# plt.figure()
# gm.field_plot_threshold(total_pot_grad_handle, 10, 30)

# plt.show()
