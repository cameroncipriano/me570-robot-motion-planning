"""
Classes and utility functions for working with graphs (plotting, search, initialization, etc.)
"""

import pickle
from math import pi, isnan
from matplotlib import pyplot as plt
import numpy as np
import me570_geometry
from me570_queue import Priority
import me570_potential as pot


def plot_arrows_from_list(arrow_list, scale=1.0, color=(0., 0., 0.)):
    """
    Plot arrows from a list of pairs of base points and displacements
    """
    x_edges, v_edges = [np.hstack(x) for x in zip(*arrow_list)]
    plt.quiver(x_edges[0, :],
               x_edges[1, :],
               v_edges[0, :],
               v_edges[1, :],
               angles='xy',
               scale_units='xy',
               scale=scale,
               color=color)


def plot_text(coord, str_label, color=(1., 1., 1.)):
    """
    Wrap plt.text to get a consistent look
    """
    plt.text(coord[0].item(),
             coord[1].item(),
             str_label,
             ha="center",
             va="center",
             fontsize='xx-small',
             bbox=dict(boxstyle="round", fc=color, ec=None))


class Graph:
    """
    A class collecting a  graph_vector data structure and all the functions that operate on a graph.
    """
    def __init__(self, graph_vector):
        """
        Stores the arguments as internal attributes.
        """
        self.graph_vector = graph_vector

    def _apply_neighbor_function(self, func):
        """
        Apply a function on each node and chain the result
        """
        list_of_lists = [func(n) for n in self.graph_vector]
        return [e for l in list_of_lists for e in l]

    def _neighbor_weights_with_positions(self, n_current):
        """
        Get all weights and where to display them
        """
        x_current = n_current['x']
        return [
            (weight_neighbor,
             self.graph_vector[idx_neighbor]['x'] * 0.25 + x_current * 0.75)
            for (weight_neighbor, idx_neighbor
                 ) in zip(n_current['neighbors_cost'], n_current['neighbors'])
        ]

    def _neighbor_displacements(self, n_current):
        """
        Get all displacements with respect to the neighbors for a given node
        """
        x_current = n_current['x']
        return [(x_current, self.graph_vector[idx_neighbor]['x'] - x_current)
                for idx_neighbor in n_current['neighbors']]

    def _neighbor_backpointers(self, n_current):
        """
        Get coordinates for backpointer arrows
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [
                (x_current,
                 0.5 * (self.graph_vector[idx_backpointer]['x'] - x_current))
            ]
        else:
            arrow = []
        return arrow

    def _neighbor_backpointers_cost(self, n_current):
        """
        Get value and coordinates for backpointer costs
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [(n_current['g'],
                      self.graph_vector[idx_backpointer]['x'] * 0.25 +
                      x_current * 0.75)]
        else:
            arrow = []
        return arrow

    def has_backpointers(self):
        """
        Return True if self.graph_vector has a "backpointer" field
        """
        return self.graph_vector is not None and len(
            self.graph_vector) > 0 and 'backpointer' in self.graph_vector[0]

    def plot(self,
             flag_edges=True,
             flag_labels=False,
             flag_edge_weights=False,
             flag_backpointers=True,
             flag_backpointers_cost=True,
             node_lists=None):
        """
        The function plots the contents of the graph described by the  graph_vector structure,
        alongside other related, optional data.
        """

        if flag_edges:
            displacement_list = self._apply_neighbor_function(
                self._neighbor_displacements)
            plot_arrows_from_list(displacement_list, scale=1.05)

        if flag_labels:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                plot_text(x_current, str(idx))

        if flag_edge_weights:
            weight_list = self._apply_neighbor_function(
                self._neighbor_weights_with_positions)
            for (weight, coord) in weight_list:
                plot_text(coord, str(weight), color=(.8, .8, 1.))

        if flag_backpointers and self.has_backpointers():
            backpointer_arrow_list = self._apply_neighbor_function(
                self._neighbor_backpointers)
            plot_arrows_from_list(backpointer_arrow_list,
                                  scale=1.05,
                                  color=(1., 0., 0.))

        if flag_backpointers_cost and self.has_backpointers:
            backpointer_cost_list = self._apply_neighbor_function(
                self._neighbor_backpointers_cost)
            for (cost, coord) in backpointer_cost_list:
                plot_text(coord, str(cost), color=(1., .8, .8))

        if node_lists is not None:
            if not isinstance(node_lists[0], list):
                node_lists = [node_lists]
            markers = ['d', 'o', 's', '*', 'h', '^', '8']
            for i, lst in enumerate(node_lists):
                x_list = [self.graph_vector[e]['x'] for e in lst]
                coords = np.hstack(x_list)
                plt.plot(
                    coords[0, :],
                    coords[1, :],
                    markers[i % len(markers)],
                    markersize=10,
                )

    def nearest_neighbors(self, x_query, k_nearest):
        """
        Returns the k nearest neighbors in the graph for a given point.
        """

        x_graph = np.hstack([n['x'] for n in self.graph_vector])
        distances_squared = np.sum((x_graph - x_query)**2, 0)
        idx = np.argpartition(distances_squared, k_nearest)
        return idx[:k_nearest]

    def heuristic(self, idx_x: int, idx_goal: int) -> float:
        """
        Computes the heuristic  h given by the Euclidean distance between the nodes with indexes
        idx_x and  idx_goal.
        """
        # Ensures the passed
        if not 0 <= idx_x < len(self.graph_vector):
            raise ValueError(
                f"idx_x passed '{idx_x}' is not a valid index in the graph")

        if not 0 <= idx_goal < len(self.graph_vector):
            raise ValueError(
                f"idx_goal passed '{idx_goal}' is not a valid index in the graph"
            )

        node_coords = self.graph_vector[idx_x]['x']
        goal_coords = self.graph_vector[idx_goal]['x']
        h_val = np.linalg.norm(goal_coords - node_coords)
        return h_val

    def get_expand_list(self, idx_n_best: int, idx_closed: list) -> list:
        """
        Finds the neighbors of element  idx_n_best that are not in  idx_closed (line   in Algorithm~
        ).
        """
        if not 0 <= idx_n_best < len(self.graph_vector):
            raise ValueError(
                f"idx_n_best '{idx_n_best}' is not a valid index in the graph")

        curr_node_neighbors = self.graph_vector[idx_n_best]["neighbors"]
        idx_expand = [
            index for index in curr_node_neighbors if index not in idx_closed
        ]
        return idx_expand

    def expand_element(self, idx_n_best: int, idx_x: int, idx_goal: int,
                       pq_open: Priority) -> Priority:
        """
        This function expands the vertex with index  idx_x (which is a neighbor of the one with
        index  idx_n_best) and returns the updated versions of  graph_vector and  pq_open.
        """

        # g is purely previous g with neighbors_cost
        # f(n) = g(n) + h(n) --> Priority Queue

        n_best_node = self.graph_vector[idx_n_best]
        x_node = self.graph_vector[idx_x]

        g_n_best = n_best_node['g']

        c_best_to_x = n_best_node['neighbors_cost'][
            n_best_node["neighbors"].index(idx_x)]

        updated_g = g_n_best + c_best_to_x
        f_n = updated_g + self.heuristic(idx_x, idx_goal)

        if not pq_open.is_member(idx_x):
            x_node['g'] = updated_g
            x_node['backpointer'] = idx_n_best
            pq_open.insert(idx_x, f_n)
        elif updated_g < x_node['g']:
            x_node['g'] = updated_g
            x_node['backpointer'] = idx_n_best

        return pq_open

    def path(self, idx_start, idx_goal):
        """
        This function follows the backpointers from the node with index  idx_goal in  graph_vector
        to the one with index  idx_start node, and returns the  coordinates (not indexes) of the
        sequence of traversed elements.
        """
        x_path = self.graph_vector[idx_goal]['x']
        curr_bp = self.graph_vector[idx_goal]['backpointer']

        while curr_bp != idx_start:
            x_path = np.hstack((self.graph_vector[curr_bp]['x'], x_path))
            curr_bp = self.graph_vector[curr_bp]['backpointer']
        # Make sure to grab the final element which will be the start vector since the while loop
        # will termiante at the first element
        x_path = np.hstack((self.graph_vector[curr_bp]['x'], x_path))

        return x_path

    def search(self, idx_start, idx_goal):
        """
        Implements the  A^* algorithm, as described by the pseudo-code in Algorithm~ .
        """
        # Initialize every node to have 0 backpointer cost and no backpointer
        for node in self.graph_vector:
            node['g'] = 0.0
            node['backpointer'] = None

        priority_queue = Priority()
        priority_queue.insert(idx_start, 0)

        idx_closed = []
        i = 0
        i_max = 100000

        while priority_queue.queue:
            idx_best, _ = priority_queue.min_extract()
            idx_closed.append(idx_best)
            if idx_best == idx_goal:
                break

            unexplored_neighbors = self.get_expand_list(idx_best, idx_closed)

            for _, neighbor_node in enumerate(unexplored_neighbors):
                priority_queue = self.expand_element(idx_best, neighbor_node,
                                                     idx_goal, priority_queue)
            if i >= i_max:
                break
            i += 1

        x_path = self.path(idx_start, idx_goal)
        return x_path

    def search_start_goal(self, x_start, x_goal):
        """
        This function performs the following operations:
         - Identifies the two indexes  idx_start,  idx_goal in  graph.graph_vector that are closest
        to  x_start and  x_goal (using Graph.nearestNeighbors twice, see Question~ -nearest).
         - Calls Graph.search to find a feasible sequence of points  x_path from  idx_start to
        idx_goal.
         - Appends  x_start and  x_goal, respectively, to the beginning and the end of the array
        x_path.
        """
        idx_start = self.nearest_neighbors(x_start, 1)[0]
        idx_goal = self.nearest_neighbors(x_goal, 1)[0]
        x_path = self.search(idx_start, idx_goal)

        return x_path


class SphereWorldGraph:
    """
    A discretized version of the  SphereWorld from Homework 3 with the addition of a search
function.
    """
    def __init__(self, nb_cells):
        """
        The function performs the following steps:
         - Instantiate an object of the class  SphereWorld from Homework 3 to load the contents of
        the file sphereworld.mat. Store the object as the internal attribute  sphereworld.d
         - Initializes an object  grid from the class  Grid initialized with arrays  xx_grid and
        yy_grid, each one containing  nb_cells values linearly spaced values from  -10 to  10.
         - Use the method grid.eval to obtain a matrix in the format expected by grid2graph in
        Question~ , i.e., with a  true if the space is free, and a  false if the space is occupied
        by a sphere at the corresponding coordinates. The quickest way to achieve this is to
        manipulate the output of Total.eval (for checking collisions with the spheres) while using
        it in conjunction with grid.eval (to evaluate the collisions along all the points on the
        grid); note that the choice of the attractive potential here does not matter.
         - Call grid2graph.
         - Store the resulting  graph object as an internal attribute.
        """
        self.world = pot.SphereWorld()
        self.grid = me570_geometry.Grid(xx_grid=np.linspace(-10, 10, nb_cells),
                                        yy_grid=np.linspace(-10, 10, nb_cells))

        potential = {
            'shape': 'conic',
            'x_goal': np.vstack(self.world.x_goal[:, 0]),
            'repulsive_weight': 10
        }
        world_potential = pot.Total(self.world, potential)

        f_handle_total = lambda point: not isnan(world_potential.eval(point))
        self.grid.eval(f_handle_total)

        self.graph = grid2graph(self.grid)

    def plot(self):
        """
        Plots the graph attribute
        """
        self.graph.plot(flag_backpointers_cost=False)

    def run_plot(self):
        """
        - Load the variables  x_start,  x_goal from the internal attribute  sphereworld.
        homework4_sphereworldPlot
        """
        for goal in self.world.x_goal.T:
            goal = goal.reshape(-1, 1)
            for start in self.world.x_start.T:
                start = start.reshape(-1, 1)
                x_path = self.graph.search_start_goal(start, goal)
                plt.plot(x_path[0, :], x_path[1, :], 'r')
            self.plot()
            plt.show()


def graph_load_test_data(variable_name):
    """
    Loads data from the file graph_test_data.pkl.
    """
    with open('graph_test_data.pkl', 'rb') as fid:
        saved_data = pickle.load(fid)
    return saved_data[variable_name]


def grid2graph(grid: me570_geometry.Grid) -> Graph:
    """
    The function returns a  Graph object described by the inputs. See Figure~  for an example of the
    expected inputs and outputs.
    """

    # Make sure values in F are logicals
    fun_evalued = np.vectorize(bool)(grid.fun_evalued)

    # Get number of columns, rows, and nodes
    nb_xx, nb_yy = fun_evalued.shape
    nb_nodes = np.sum(fun_evalued)

    # Get indeces of non-zero entries, and assign a progressive number to each
    idx_xx, idx_yy = np.where(fun_evalued)
    idx_assignment = range(0, nb_nodes)

    # Lookup table from xx,yy element to assigned index (-1 means not assigned)
    idx_lookup = -1 * np.ones(fun_evalued.shape, 'int')
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        idx_lookup[i_xx, i_yy] = i_assigned

    def grid2graph_neighbors(idx_xx, idx_yy):
        """
        Finds the neighbors of a given element
        """

        displacements = [(idx_xx + dx, idx_yy + dy) for dx in [-1, 0, 1]
                         for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        neighbors = []
        for i_xx, i_yy in displacements:
            if 0 <= i_xx < nb_xx and 0 <= i_yy < nb_yy and idx_lookup[
                    i_xx, i_yy] >= 0:
                neighbors.append(idx_lookup[i_xx, i_yy].item())

        return neighbors

    # Create graph_vector data structure and populate 'x' and 'neighbors' fields
    graph_vector = [None] * nb_nodes
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        x_current = np.array([[grid.xx_grid[i_xx]], [grid.yy_grid[i_yy]]])
        neighbors = grid2graph_neighbors(i_xx, i_yy)
        graph_vector[i_assigned] = {'x': x_current, 'neighbors': neighbors}

    # Populate the 'neighbors_cost' field
    # Cannot be done in the loop above because not all 'x' fields would be filled
    for idx, n_current in enumerate(graph_vector):
        x_current = n_current['x']
        if not n_current['neighbors']:
            neighbors_cost_np = []
        else:
            x_neighbors = np.hstack(
                [graph_vector[idx]['x'] for idx in n_current['neighbors']])
            neighbors_cost_np = np.sum((x_neighbors - x_current)**2, 0)
        graph_vector[idx]['neighbors_cost'] = list(neighbors_cost_np)

    return Graph(graph_vector)


def test_nearest_neighbors():
    """
    Tests Graph.nearest_neighbors by picking a random point and finding the 3 nearest neighbors
    in graphVectorMedium
    """
    graph = Graph(graph_load_test_data('graphVectorMedium'))
    x_query = np.array([[5], [4]]) * np.random.rand(2, 1)
    idx_neighbors = graph.nearest_neighbors(x_query, 3)
    graph.plot(node_lists=idx_neighbors)
    plt.scatter(x_query[[0]], x_query[[1]])


def test_grid2graph():
    """
    Tests grid2graph() by creating an arbitrary function returning bools
    """
    xx_grid = np.linspace(0, 2 * pi, 40)
    yy_grid = np.linspace(0, pi, 20)
    func = lambda x: (x[[1]] > pi / 2 or np.linalg.norm(x - np.ones(
        (2, 1))) < 0.75) and not np.linalg.norm(x - np.array([[4], [2.5]])
                                                ) < 0.5
    grid = me570_geometry.Grid(xx_grid, yy_grid)
    grid.eval(func)
    graph = grid2graph(grid)
    graph.plot()
