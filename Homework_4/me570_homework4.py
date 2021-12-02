import me570_graph as graph
import matplotlib.pyplot as plt


def main():
    plt.figure()
    # graph.Graph(graph.graph_load_test_data("closedMedium")).plot()
    # plt.show()
    graph_vector = graph.graph_load_test_data("graphVector")
    print(graph_vector)

    # graph.Graph(graph.graph_load_test_data("graphVector")).plot()
    # plt.show()
    # graph.Graph(graph.graph_load_test_data("graphVectorMedium")).plot()
    # plt.show()
    # graph.Graph(graph.graph_load_test_data("graphVectorMedium_solved")).plot()
    # plt.show()
    # graph.Graph(graph.graph_load_test_data("graphVector_solved")).plot()
    # plt.show()


if __name__ == "__main__":
    main()