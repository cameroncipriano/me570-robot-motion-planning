"""
Combine the classes below with the file me570_robot.py from previous assignments
"""

from scipy import io as scio
import me570_geometry


class TwoLinkGraph:
    """
    A class for finding a path for the two-link manipulator among given obstacle points using a grid
discretization and  A^*.
    """
    def load_free_space_graph(self):
        """
        The function performs the following steps
         - Calls the method load_free_space_grid.
         - Calls grid2graph.
         - Stores the resulting  graph object of class  Grid as an internal attribute.
        """
        pass  # Substitute with your code

    def plot(self):
        """
        Use the method Graph.plot to visualize the contents of the attribute  graph.
        """
        pass  # Substitute with your code

    def search_start_goal(self, theta_start, theta_goal):
        """
        Use the method Graph.search to search a path in the graph stored in  graph.
        """
        pass  # Substitute with your code
        return theta_path


def load_free_space_grid():
    """
Loads the contents of the file ! twolink_freeSpace_data.mat
    """
    test_data = scio.loadmat('twolink_freeSpace_data.mat')
    test_data = test_data['grid'][0][0]
    grid = me570_geometry.Grid(test_data[0], test_data[1])
    grid.fun_evalued = test_data[2]
    return grid
