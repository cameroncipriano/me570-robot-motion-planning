"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import me570_geometry as gm


class TwoLink:
    """
    Class for creating our Two_Link Manipulator
    """
    def __init__(self) -> None:
        """
        Creates the two polygons necessary for the robot
        """
        add_y_reflection = lambda vertices: np.hstack(
            [vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])

        vertices1 = np.array([[0, 5], [-1.11, -0.511]])
        vertices1 = add_y_reflection(vertices1)
        vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                              [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
        vertices2 = add_y_reflection(vertices2)
        self._polygons = (gm.Polygon(vertices1), gm.Polygon(vertices2))

    def polygons(self):
        """
        Returns two polygons that represent the links in a simple 2-D two-link manipulator.
        """
        return self._polygons

    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
    transformed according to  _1, _2.
        """

        # Rotation matrices
        w_r_beta_1 = gm.rot2d(theta[0, 0])
        beta_1_r_beta_2 = gm.rot2d(theta[1, 0])
        w_r_beta_2 = w_r_beta_1 @ beta_1_r_beta_2

        # Translation matrix
        beta_1_t_beta_2 = np.vstack((5, 0))
        w_t_beta_2 = gm.rot2d(theta[0, 0]) @ beta_1_t_beta_2

        # Transform End effector from β₂ to the world
        vertex_effector_transf = (w_r_beta_2 @ np.vstack((5, 0))) + w_t_beta_2

        # Polygon1's coordinates are in the β₁ coordinate space, so we need to calculate ʷp_β₁
        polygon_1_vert_transf = np.array([[], []])
        for i in range(self._polygons[0].nb_vertices):
            vertex = np.vstack(self._polygons[0].vertices[:, i])
            vertex_transf = (w_r_beta_1 @ vertex)
            polygon_1_vert_transf = np.hstack(
                (polygon_1_vert_transf, vertex_transf))

        polygon1_transf = gm.Polygon(polygon_1_vert_transf)

        # Polygon2's coordinates are in the β₂ coordinate space, so we need to calculate ʷp_β₂
        polygon_2_vert_transf = np.array([[], []])
        for i in range(self._polygons[1].nb_vertices):
            vertex = np.vstack(self._polygons[1].vertices[:, i])
            vertex_transf = (w_r_beta_2 @ vertex) + w_t_beta_2
            polygon_2_vert_transf = np.hstack(
                (polygon_2_vert_transf, vertex_transf))

        polygon2_transf = gm.Polygon(polygon_2_vert_transf)

        return vertex_effector_transf, polygon1_transf, polygon2_transf

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color)
        polygon2_transf.plot(color)

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """

        flag_theta = [False] * theta.shape[1]

        for i in range(theta.shape[1]):
            config = np.vstack(theta[:, i])
            [_, polygon1_transf, polygon2_transf] = self.kinematic_map(config)

            flag_points = polygon1_transf.is_collision(points)
            # Must logically reverse the array because Polygon.is_collision is
            # returning the incorrect (opposite) answer
            if True in np.logical_not(flag_points):
                flag_theta[i] = True

            flag_points = polygon2_transf.is_collision(points)
            # Must logically reverse the array because Polygon.is_collision is
            # returning the incorrect (opposite) answer
            if True in np.logical_not(flag_points):
                flag_theta[i] = True

        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """
        collisions = self.is_collision(theta, points)
        for i, is_collision in enumerate(collisions):
            if is_collision:
                color = 'r'
            else:
                color = 'g'

            self.plot(np.vstack(theta[:, i]), color)
            plt.scatter(points[0, :], points[1, :], c='k', marker='*')

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """
        vertex_effector_dot = []

        for i in range(theta.shape[1]):
            curr_theta = theta[:, i]
            curr_theta_dot = np.vstack(theta_dot[:, i])

            sin_theta_1 = math.sin(curr_theta[0])
            cos_theta_1 = math.cos(curr_theta[0])

            sin_theta_2 = math.sin(curr_theta[1])
            cos_theta_2 = math.cos(curr_theta[1])

            derivative_at_point = (5 * np.array(
                [[
                    -sin_theta_1 * cos_theta_2 - cos_theta_1 * sin_theta_2 -
                    sin_theta_1, -cos_theta_1 * sin_theta_2 -
                    sin_theta_1 * cos_theta_2 + cos_theta_1
                ],
                 [
                     cos_theta_1 * cos_theta_2 - sin_theta_1 * sin_theta_2 +
                     cos_theta_1, -sin_theta_1 * sin_theta_2 +
                     cos_theta_1 * cos_theta_2 + sin_theta_1
                 ]])) @ curr_theta_dot

            vertex_effector_dot.append(derivative_at_point)

        return vertex_effector_dot
