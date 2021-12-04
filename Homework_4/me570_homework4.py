from math import pi
import matplotlib.pyplot as plt
from scipy import io as scio
import numpy as np
import me570_graph as graph
import me570_robot as robot
from me570_queue import Priority
from me570_graph import SphereWorldGraph


def main():
    plt.figure()

    # my_graph = graph.Graph(graph.graph_load_test_data("graphVectorMedium"))
    # # my_graph.plot(flag_labels=True,
    # #               flag_edge_weights=True,
    # #               flag_backpointers=True,
    # #               flag_backpointers_cost=True)

    # # Expand List tests
    # # print(f"expanded_list from 0 : {my_graph.get_expand_list(0, [])}")
    # # print(f"expanded_list from 0 : {my_graph.get_expand_list(0, [1])}")

    # # Testing A* Algorithm
    # x_path = my_graph.search(0, 13)
    # print(x_path)

    # my_graph.plot(flag_labels=True,
    #               flag_edge_weights=False,
    #               flag_backpointers=True,
    #               flag_backpointers_cost=True)

    # plt.show()

    # graph.Graph(graph.graph_load_test_data("graphVectorMedium_solved")).plot()
    # plt.show()

    sphere_world = SphereWorldGraph(20)
    # sphere_world.plot()
    # plt.show()
    # sphere_world.run_plot()

    test_data = scio.loadmat("twolink_testData.mat")
    test_points = test_data['obstaclePoints']

    # theta = np.vstack((0, 0))
    # for angle in np.linspace(0, 2 * pi, 15):
    #     theta = np.hstack((theta, np.vstack((angle, pi / 7))))

    my_robot = robot.TwoLink()
    my_robot_graph = robot.TwoLinkGraph()
    my_robot_graph.plot()
    # plt.scatter(test_points[0, :],
    #             test_points[1, :],
    #             c='r',
    #             marker='x',
    #             linewidths=0.5)

    # Easy case
    # theta_start = np.vstack((0.76, 0.12))
    # theta_goal = np.vstack((0.76, 6.00))

    # Medium Case
    theta_start = np.vstack((0.76, 0.12))
    theta_goal = np.vstack((2.72, 5.45))

    theta_path = my_robot_graph.search_start_goal(theta_start, theta_goal)
    plt.plot(theta_path[0, :], theta_path[1, :], 'r')
    my_robot_graph.plot()

    # my_robot.plot_animate(theta_path)
    plt.show()


if __name__ == "__main__":
    main()