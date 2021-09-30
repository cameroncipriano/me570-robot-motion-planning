"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import numpy as np
import me570_geometry as gm


class TwoLink:
    def __init__(self) -> None:
        add_y_reflection = lambda vertices: np.hstack(
            [vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])

        vertices1 = np.array([[0, 5], [-1.11, -0.511]])
        vertices1 = add_y_reflection(vertices1)
        vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                              [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
        vertices2 = add_y_reflection(vertices2)
        self.Polygons = (gm.Polygon(vertices1), gm.Polygon(vertices2))

    def polygons(self):
        """
        Returns two polygons that represent the links in a simple 2-D two-link manipulator.
        """
        return self.Polygons

    """ This class was introduced in a previous homework. """

    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
    transformed according to  _1, _2.
        """

        theta_1 = theta[0, 0]
        theta_2 = theta[1, 0]

        # Rotation matrices
        w_R_beta_1 = gm.rot2d(theta_1)
        beta_1_R_beta_2 = gm.rot2d(theta_2)
        w_R_beta_2 = w_R_beta_1 * beta_1_R_beta_2

        # Translation matrix
        beta_1_T_beta_2 = np.vstack((5, 0))

        # Polygon1's coordinates are in the β₁ coordinate space, so we need to calculate ʷp_β₁
        for i in range(self.Polygons[0].nb_vertices):
            continue

        # for each vertex in the polygon
        w_p_beta_1 = w_R_beta_1

        # Polygon2's coordinates are in the β₂coordinate space
        return vertex_effector_transf, polygon1_transf, polygon2_transf

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [vertex_effector_transf, polygon1_transf,
         polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color)
        polygon2_transf.plot(color)

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """
        pass  # Substitute with your code
        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """
        pass  # Substitute with your code

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """
        pass  # Substitute with your code
        return vertex_effector_dot
