import me570_graph as graph
import me570_robot as robot
import matplotlib.pyplot as plt
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
    sphere_world.run_plot()
    plt.show()


if __name__ == "__main__":
    main()