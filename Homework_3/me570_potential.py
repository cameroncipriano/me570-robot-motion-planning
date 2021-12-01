"""
Classes to define potential and potential planner for the sphere world
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scio
import me570_geometry
from me570_qp import qp_supervisor


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """
        data = scio.loadmat('sphereWorld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = np.asscalar(sphere_args[1])
            sphere_args[2] = np.asscalar(sphere_args[2])
            self.world.append(me570_geometry.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a
        * marker at the goal location.
        """

        for sphere in self.world:
            sphere.plot('k')

        plt.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')

        plt.xlim([-11, 11])
        plt.ylim([-11, 11])


class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """
    Evaluate the repulsive potential from  sphere at the location x= x_eval. The function returns
    the repulsive potential as given by
    (  eq:repulsive  ).
        """
        d_subi_x = self.sphere.distance(x_eval)

        if d_subi_x > self.sphere.distance_influence:
            u_rep = 0
        elif 0 < d_subi_x < self.sphere.distance_influence:
            u_rep = 0.5 * (1 / d_subi_x -
                           1 / self.sphere.distance_influence)**2
        else:
            u_rep = math.inf

        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by
        (eq:repulsive-gradient).
        """
        d_subi_x = self.sphere.distance(x_eval)
        grad_d_subi_x = self.sphere.distance_grad(x_eval)

        if d_subi_x > self.sphere.distance_influence:
            grad_u_rep = 0
        elif 0 < d_subi_x < self.sphere.distance_influence:
            grad_u_rep = -(1 / d_subi_x - 1 / self.sphere.distance_influence
                           ) * (1 / d_subi_x**2) * grad_d_subi_x
        else:
            grad_u_rep = math.nan

        return grad_u_rep


class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        # Test if the necessary keys are a subset of self.potential
        if {'shape', 'x_goal'} <= self.potential.keys():
            attr_shape = self.potential['shape'].lower()
            if attr_shape == "conic":
                u_attr = np.linalg.norm(x_eval - self.potential['x_goal'])
            elif attr_shape == "quadratic":
                u_attr = np.linalg.norm(x_eval - self.potential['x_goal'])**2
            else:
                raise NotImplementedError(
                    f"Attractive Potential is undefined for shape, \"{attr_shape}\""
                )
        else:
            raise ValueError(
                'Definition of potential does not indicate a shape')

        return u_attr

    def grad(self, x_eval):
        """
    Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient is
    given by the formula If  potential['shape'] is equal to  'conic', use p=1; if it is equal to
    'quadratic', use p=2.
        """
        # Test if the necessary keys are a subset of self.potential
        if {'shape', 'x_goal'} <= self.potential.keys():
            attr_shape = self.potential['shape'].lower()
            if attr_shape == "conic":
                grad_u_attr = (x_eval - self.potential['x_goal']
                               ) / np.linalg.norm(x_eval -
                                                  self.potential['x_goal'])
            elif attr_shape == "quadratic":
                grad_u_attr = x_eval - self.potential['x_goal']
            else:
                raise NotImplementedError(
                    f"Attractive Potential is undefined for shape, \"{attr_shape}\""
                )
        else:
            raise ValueError(
                'Definition of potential does not indicate a shape')
        return grad_u_attr


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world: SphereWorld = world
        self.potential: dict = potential

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        # Ensure the proper fields are present
        # obstacles: list[RepulsiveSphere] = []
        obstacle_potential = 0
        if {'shape', 'x_goal', 'repulsive_weight'} <= self.potential.keys():
            attr = Attractive(self.potential)
            for sphere in self.world.world:
                obstacle_potential = obstacle_potential + RepulsiveSphere(
                    sphere).eval(x_eval)
        else:
            raise ValueError(
                "Must have all necessary fields: 'shape', 'x_goal', and 'repulsive_weight'"
            )

        u_eval = attr.eval(
            x_eval) + self.potential['repulsive_weight'] * obstacle_potential

        return u_eval

    def grad(self, x_eval):
        """
    Compute the gradient of the total potential,
    U= U_ attr+    _i U_ rep,i, where   is given by the variable
    potential.repulsiveWeight
        """
        # Ensure the proper fields are present
        obstacle_gradient = np.zeros((2, 1))
        if {'shape', 'x_goal', 'repulsive_weight'} <= self.potential.keys():
            attr = Attractive(self.potential)
            for sphere in self.world.world:
                obstacle_gradient = obstacle_gradient + RepulsiveSphere(
                    sphere).grad(x_eval)
        else:
            raise ValueError(
                "Must have all necessary fields: 'shape', 'x_goal', and 'repulsive_weight'"
            )

        grad_u_eval = attr.grad(
            x_eval) + self.potential['repulsive_weight'] * obstacle_gradient
        return grad_u_eval


class Planner:
    """
    Planner for creating the path from start -> goal
    """
    def run(self, x_start, planner_parameters):
        """
        This function uses a given function ( planner_parameters['control']) to implement a generic
    potential-based planner with step size  planner_parameters['epsilon'], and evaluates the cost
    along the returned path. The planner must stop when either the number of steps given by
    planner_parameters['nb_steps'] is reached, or when the norm of the vector given by
    planner_parameters['control'] is less than 5 10^-3 (equivalently,  5e-3).
        """

        # Create the trajectory path of the robot, pre-filling the list with 0s
        # in case the planner stops early
        x_path = np.zeros((2, planner_parameters['nb_steps']))

        # The first value is always going to be where the robot starts
        x_path[:, 0] = x_start.T

        # Create the list of potential values, pre-filling the list with 0s in case
        # the planner stops early
        u_path = np.zeros(planner_parameters['nb_steps'])

        # The first potential is always going to be evaluated at the start location
        u_path[0] = planner_parameters['U'](x_start)

        for k in range(planner_parameters['nb_steps'] - 1):
            x_val = np.vstack(x_path[:, k])
            control_val = planner_parameters['control'](x_val)

            # Determines if the planner should stop since the gradient is essentially 0
            # (meaning either stuck or found goal)
            if np.linalg.norm(control_val) < 5e-3:
                x_path[:, k:] = math.nan
                u_path[k:] = math.nan
                break

            x_path[:, k + 1] = (x_val +
                                planner_parameters['epsilon'] * control_val).T

            u_path[k + 1] = planner_parameters['U'](x_val)
        return x_path, u_path

    def run_plot(self):
        """
        This function performs the following steps:
     - Loads the problem data from the file !70!DarkSeaGreen2 sphereworld.mat.
     - For each goal location in  world.xGoal:
     - Uses the function Sphereworld.plot to plot the world in a first figure.
     - Sets  planner_parameters['U'] to the negative of  Total.grad.
     - it:grad-handle Calls the function Potential.planner with the problem data and the input
    arguments. The function needs to be called five times, using each one of the initial locations
    given in  x_start (also provided in !70!DarkSeaGreen2 sphereworld.mat).
     - it:plot-plan After each call, plot the resulting trajectory superimposed to the world in the
    first subplot; in a second subplot, show  u_path (using the same color and using the  semilogy
    command).
        """
        world = SphereWorld()
        # self.plot_quadratics(
        #     {
        #         'repulsive_weight': 35,
        #         'epsilon': 2e-2,
        #         'nb_steps': 600
        #     }, world)
        # self.plot_conics(
        #     {
        #         'repulsive_weight': 1,
        #         'epsilon': 3.55e-1,
        #         'nb_steps': 1000
        #     }, world)
        self.plot_quad_clfcbf(
            {
                'repulsive_weight': 5,
                'epsilon': 6.0e-2,
                'nb_steps': 100
            }, world)

        plt.show()

    def plot_conics(self, config, world):
        """
        Method used for plotting conic potential functions
        """
        plt.rcParams["figure.figsize"] = (8, 8)
        _, axs = plt.subplots(world.x_goal.shape[1], 2)
        colors = plt.cm.get_cmap('hsv', world.x_start.shape[1] + 1)

        total_potential = None
        planner_parameters = None
        plt.subplots_adjust(hspace=0.305)

        for i, loc in enumerate(world.x_goal.T):

            # Setting up the constants and parameters we need
            total_potential = Total(
                world, {
                    'x_goal': np.vstack(loc),
                    'shape': 'conic',
                    'repulsive_weight': config['repulsive_weight']
                })
            planner_parameters = {
                'U': lambda point: total_potential.eval(point),
                'control': lambda point: -total_potential.grad(point),
                'epsilon': config['epsilon'],
                'nb_steps': config['nb_steps']
            }

            for color_num, start_loc in enumerate(world.x_start.T):
                curr_start = np.vstack(start_loc)
                x_path, u_path = self.run(curr_start, planner_parameters)
                # Make sure we are plotting below the world we are concerned with
                plt.sca(axs[i, 0])
                plt.plot(x_path[0, :], x_path[1, :], color=colors(color_num))

                # Plotting the potential on the right-hand subplot
                plt.sca(axs[i, 1])
                plt.title(f"Goal {i}, Potential Conic")
                plt.xlabel('# steps')
                plt.ylabel('U')
                plt.semilogy(np.arange(0, planner_parameters['nb_steps']),
                             u_path,
                             color=colors(color_num))

            # Set current axis to left column to draw the world
            plt.sca(axs[i, 0])
            plt.title(f"Goal {i}, Trajectories Conic")
            world.plot()

    def plot_quadratics(self, config, world):
        """
        Method for plotting quadratic potential functions
        """
        plt.rcParams["figure.figsize"] = (8, 8)
        _, axs = plt.subplots(world.x_goal.shape[1], 2)
        colors = plt.cm.get_cmap('hsv', world.x_start.shape[1] + 1)

        plt.subplots_adjust(hspace=0.305)

        total_potential = None
        planner_parameters = None
        for i, loc in enumerate(world.x_goal.T):

            # Setting up the constants and parameters we need
            total_potential = Total(
                world, {
                    'x_goal': np.vstack(loc),
                    'shape': 'quadratic',
                    'repulsive_weight': config['repulsive_weight']
                })
            planner_parameters = {
                'U': lambda point: total_potential.eval(point),
                'control': lambda point: -total_potential.grad(point),
                'epsilon': config['epsilon'],
                'nb_steps': config['nb_steps']
            }

            for color_num, start_loc in enumerate(world.x_start.T):
                curr_start = np.vstack(start_loc)
                x_path, u_path = self.run(curr_start, planner_parameters)
                # Make sure we are plotting below the world we are concerned with
                plt.sca(axs[i, 0])
                plt.plot(x_path[0, :], x_path[1, :], color=colors(color_num))

                # Plotting the potential on the right-hand subplot
                plt.sca(axs[i, 1])
                plt.title(f"Goal {i}, Potential Quadratic")
                plt.xlabel('# steps')
                plt.ylabel('U')
                plt.semilogy(np.arange(0, planner_parameters['nb_steps']),
                             u_path,
                             color=colors(color_num))

            # Set current axis to left column to draw the world
            plt.sca(axs[i, 0])
            plt.title(f"Goal {i}, Trajectories Quadratic")
            world.plot()

    def plot_quad_clfcbf(self, config, world):
        """
        Method for plotting quadratic potential functions using CLF-CBF Formulation
        """
        plt.rcParams["figure.figsize"] = (8, 8)
        _, axs = plt.subplots(world.x_goal.shape[1], 2)
        colors = plt.cm.get_cmap('hsv', world.x_start.shape[1] + 1)

        plt.subplots_adjust(hspace=0.305)

        # Get rid of pylint errors of defining a variable in a loop
        potential = None
        planner_parameters = None
        for i, loc in enumerate(world.x_goal.T):

            # Setting up the constants and parameters we need
            potential = {
                'x_goal': np.vstack(loc),
                'shape': 'quadratic',
                'repulsive_weight': config['repulsive_weight']
            }
            planner_parameters = {
                'U': lambda point: Total(world, potential).eval(point),
                'control':
                lambda point: clfcbf_control(point, world, potential),
                'epsilon': config['epsilon'],
                'nb_steps': config['nb_steps']
            }

            for color_num, start_loc in enumerate(world.x_start.T):
                curr_start = np.vstack(start_loc)
                x_path, u_path = self.run(curr_start, planner_parameters)
                # Make sure we are plotting below the world we are concerned with
                plt.sca(axs[i, 0])
                plt.plot(x_path[0, :], x_path[1, :], color=colors(color_num))

                # Plotting the potential on the right-hand subplot
                plt.sca(axs[i, 1])
                plt.title(f"Goal {i}, Potential Quadratic")
                plt.xlabel('# steps')
                plt.ylabel('U')
                plt.semilogy(np.arange(0, planner_parameters['nb_steps']),
                             u_path,
                             color=colors(color_num))

            # Set current axis to left column to draw the world
            plt.sca(axs[i, 0])
            plt.title(f"Goal {i}, Trajectories Quadratic")
            world.plot()


def clfcbf_control(x_eval, world, potential):
    """
    Compute u^* according to
    (  eq:clfcbf-qp  ).
    """
    nb_obstacles = len(world.world)
    a_barrier = np.zeros((nb_obstacles, 2))
    b_barrier = np.zeros((nb_obstacles, 1))
    u_ref = -(Attractive(potential).grad(x_eval))

    for obst_num, sphere in enumerate(world.world):
        a_barrier[obst_num, :] = -(sphere.distance_grad(x_eval)).T
        b_barrier[obst_num, 0] = -(potential['repulsive_weight'] *
                                   sphere.distance(x_eval))

    u_opt = qp_supervisor(a_barrier, b_barrier, u_ref)
    return u_opt
