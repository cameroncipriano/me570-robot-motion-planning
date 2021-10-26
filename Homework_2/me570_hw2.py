"""
Main file for ME570 HW2
"""

import math
from scipy import io as scio
import numpy as np
import matplotlib.pyplot as plt
import me570_geometry
import me570_robot


def twolink_plot_collision_test():
    """
    This function generates 30 random configurations, loads the  points variable from the file
!70!DarkSeaGreen2 twolink_testData.mat (provided with the homework), and then display the results
using  twolink_plotCollision to plot the manipulator in red if it is in collision, and green
otherwise.
    """
    nb_configurations = 30
    two_link = me570_robot.TwoLink()
    theta_random = 2 * math.pi * np.random.rand(2, nb_configurations)
    test_data = scio.loadmat('twolink_testData.mat')
    obstacle_points = test_data['obstaclePoints']
    plt.plot(obstacle_points[0, :], obstacle_points[1, :], 'r*')
    for i_theta in range(0, nb_configurations):
        theta = theta_random[:, i_theta:i_theta + 1]
        two_link.plot_collision(theta, obstacle_points)

    plt.show()


def grid_eval_example():
    """ Example of the use of Grid.mesh and Grid.eval functions"""
    fun = lambda x: math.sin(x[0])
    example_grid = me570_geometry.Grid(np.linspace(-3, 3), np.linspace(-3, 3))
    fun_eval = example_grid.eval(fun)
    [xx_grid, yy_grid] = example_grid.mesh()
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.plot_surface(xx_grid, yy_grid, fun_eval)
    plt.show()


def torus_twolink_plot_jacobian():
    """
    For each one of the curves used in Question~ q:torusDrawChartsCurves, do the following:
 - Use Line.linspace to compute the array  thetaPoints for the curve;
 - For each one of the configurations given by the columns of  thetaPoints:
 - Use Twolink.plot to plot the two-link manipulator.
 - Use Twolink.jacobian to compute the velocity of the end effector, and then use quiver to draw
that velocity as an arrow starting from the end effector's position.   The function should produce a
total of four windows (or, alternatively, a single window with four subplots), each window (or
subplot) showing all the configurations of the manipulator superimposed on each other. You can use
matplotlib.pyplot.ion and insert a time.sleep command in the loop for drawing the manipulator, in
order to obtain a ``movie-like'' presentation of the motion.
    """
    return 1
