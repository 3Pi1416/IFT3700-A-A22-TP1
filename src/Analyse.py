from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.algos.BinaryPartitioning import calculate_binary_partition
from src.algos.Isomap import calculate_isomap
from src.algos.KMedoids import calculate_k_medoids
from src.algos.KNeighbour import evaluate_k_neighbour
from src.algos.PCoA import calculate_PCoA
from src.caculateError import calculate_error
from src.plotResult import plot_result


# a faire passer avant


def analyse_Similarity(dissimilarities, data_point_y, dissimilarities_test,
                       data_point_y_test, initial_medoids, colors, real_value_name):
    data_set: dict = {
        "PCoA": ("PCoA", calculate_PCoA(dissimilarities, dissimilarities_test)),
        "neighbour_2": ("neighbour", evaluate_k_neighbour(2, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "neighbour_3": ("neighbour", evaluate_k_neighbour(3, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "neighbour_4": ("neighbour", evaluate_k_neighbour(4, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "neighbour_5": ("neighbour", evaluate_k_neighbour(5, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "neighbour_6": ("neighbour", evaluate_k_neighbour(6, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "neighbour_7": ("neighbour", evaluate_k_neighbour(7, data_point_y.to_numpy(), dissimilarities,
                                                          dissimilarities_test)),
        "isomap_2": ("isomap", calculate_isomap(2, 1, dissimilarities, dissimilarities_test)),
        "k_medoids": ("k_medoids", calculate_k_medoids(dissimilarities, dissimilarities_test, initial_medoids)),
        "binary_partition": ("binary_partition", calculate_binary_partition(2, dissimilarities, dissimilarities_test))
    }
    dict_method = {"PCoA": None, "neighbour": [2], "isomap": None, "k_medoids": [real_value_name],
                   "binary_partition": [real_value_name]}

    test_size = 0.8

    new_data_set = {}
    new_data_point_y = []
    new_data_point_y_test = []

    for name, args in data_set.items():
        if dict_method[args[0]] is None:
            dissimilarities, dissimilarities_test, new_data_point_y, new_data_point_y_test = train_test_split(
                args[1], data_point_y_test, test_size=test_size, shuffle=False)

            new_data_set[f"{args[0]}_neighbour_2"] = (
                "neighbour", evaluate_k_neighbour(2, new_data_point_y.to_numpy(), dissimilarities, dissimilarities_test,
                                                  metric="minkowski"))
            new_data_set[f"{args[0]}_neighbour_3"] = (
                "neighbour", evaluate_k_neighbour(2, new_data_point_y.to_numpy(), dissimilarities, dissimilarities_test,
                                                  metric="minkowski"))
            new_data_set[f"{args[0]}_neighbour_4"] = (
                "neighbour", evaluate_k_neighbour(2, new_data_point_y.to_numpy(), dissimilarities, dissimilarities_test,
                                                  metric="minkowski"))
            new_data_set[f"{args[0]}_neighbour_5"] = (
                "neighbour", evaluate_k_neighbour(2, new_data_point_y.to_numpy(), dissimilarities, dissimilarities_test,
                                                  metric="minkowski"))

    new_color = colors[len(new_data_point_y):]

    calculate_error(data_set, data_point_y_test, real_value_name, dict_method)
    plot_result(data_set, colors)

    calculate_error(new_data_set, new_data_point_y_test, real_value_name, dict_method)
    plot_result(new_data_set, new_color)

    return
