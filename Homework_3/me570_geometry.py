"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import numbers
import numpy as np
from matplotlib import cm, pyplot as plt


def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical
    types

    [This function is the same as the one from HW2]
    """
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        breakpoint()
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size


class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate arbitrary
    functions on those points.

    [This class is the same as the one from HW2]
    """
    def __init__(self, xx_grid, yy_grid):
        """
        Stores the input arguments in attributes.
        """
        self.xx_grid = xx_grid
        self.yy_grid = yy_grid

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
        on each point defined by the grid.
        """

        dim_domain = [numel(self.xx_grid), numel(self.yy_grid)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_grid[idx_x]],
                                   [self.yy_grid[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])

        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """

        return np.meshgrid(self.xx_grid, self.yy_grid)


def clip(val, threshold):
    """
    If val is a scalar, threshold its value; if it is a vector, normalized it
    """
    if isinstance(val, np.ndarray):
        val_norm = np.linalg.norm(val)
        if val_norm > threshold:
            val /= val_norm
    elif isinstance(val, numbers.Number):
        if val > threshold:
            val = threshold
    else:
        raise ValueError('Numeric format not recognized')

    return val


def field_plot_threshold(f_handle, threshold=10, nb_grid=61):
    """
    The function evaluates the function  f_handle on points placed on a regular grid.
    """

    xx_grid = np.linspace(-11, 11, nb_grid)
    yy_grid = np.linspace(-11, 11, nb_grid)
    grid = Grid(xx_grid, yy_grid)

    f_handle_clip = lambda val: clip(f_handle(val), threshold)
    f_eval = grid.eval(f_handle_clip)

    [xx_mesh, yy_mesh] = grid.mesh()
    f_dim = numel(f_handle_clip(np.zeros((2, 1))))
    if f_dim == 1:
        # scalar field
        fig = plt.gcf()
        axis = fig.add_subplot(111, projection='3d')

        axis.plot_surface(xx_mesh,
                          yy_mesh,
                          f_eval.transpose(),
                          cmap=cm.gnuplot2)
        axis.view_init(90, -90)
    elif f_dim == 2:
        # vector field
        # grid.eval gives the result transposed with respect to what meshgrid expects
        f_eval = f_eval.transpose((1, 0, 2))
        # vector field
        plt.quiver(xx_mesh,
                   yy_mesh,
                   f_eval[:, :, 0],
                   f_eval[:, :, 1],
                   angles='xy',
                   scale_units='xy')
    else:
        raise NotImplementedError(
            'Field plotting for dimension greater than two not implemented')

    plt.xlabel('x')
    plt.ylabel('y')


class Sphere:
    """ Class for plotting and computing distances to spheres (circles, in 2-D). """
    def __init__(self, center, radius, distance_influence):
        """
        Save the parameters describing the sphere as internal attributes.
        """
        self.center = center
        self.radius = radius
        self.distance_influence = distance_influence

    def plot(self, color):
        """
        This function draws the sphere (i.e., a circle) of the given radius, and the specified color,
    and then draws another circle in gray with radius equal to the distance of influence.
        """
        # Get current axes
        ax = plt.gca()
        # Add circle as a patch
        if self.radius > 0:
            # Circle is filled in
            kwargs = {'facecolor': (0.3, 0.3, 0.3)}
            radius_influence = self.radius + self.distance_influence
        else:
            # Circle is hollow
            kwargs = {'fill': False}
            radius_influence = -self.radius - self.distance_influence

        center = (self.center[0, 0], self.center[1, 0])
        ax.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor=color,
                       **kwargs))

        ax.add_patch(
            plt.Circle(center,
                       radius=radius_influence,
                       edgecolor=(0.7, 0.7, 0.7),
                       fill=False))

    def distance(self, points):
        """
        Computes the signed distance between points and the sphere, while taking into account whether
    the sphere is hollow or filled in.
        """
        pass  # Substitute with your code
        return d_points_sphere

    def distance_grad(self, sphere, points):
        """
        Computes the gradient of the signed distance between points and the sphere, consistently with
    the definition of Sphere.distance.
        """
        pass  # Substitute with your code
        return grad_d_points_sphere
