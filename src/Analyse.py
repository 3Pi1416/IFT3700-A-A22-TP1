from sklearn import metrics

from src.algos.BinaryPartitioning import calculate_binary_partition
from src.algos.Isomap import calculate_isomap
from src.algos.KMedoids import calculate_k_medoids
from src.algos.KNeighbour import evaluate_k_neighbour
from src.algos.PCoA import calculate_PCoA
from src.caculateError import calculate_error
from src.plotResult import plot_result


# a faire passer avant


def analyse_Similarity(dissimilarities, data_point_y, dissimilarities_test,
                       data_point_y_test, initial_medoids, colors):
    data_set: dict = {"PCoA": ("PCoA", calculate_PCoA(dissimilarities, dissimilarities_test)),
                      "neighbour_2": ("neighbour", evaluate_k_neighbour(2, data_point_y.to_numpy(), dissimilarities,
                                                                        dissimilarities_test)),
                      "neighbour_3": ("neighbour", evaluate_k_neighbour(3, data_point_y.to_numpy(), dissimilarities,
                                                                        dissimilarities_test)),
                      "neighbour_4": ("neighbour", evaluate_k_neighbour(4, data_point_y.to_numpy(), dissimilarities,
                                                                        dissimilarities_test)),
                      "neighbour_5": ("neighbour", evaluate_k_neighbour(5, data_point_y.to_numpy(), dissimilarities,
                                                                        dissimilarities_test)),
                      "isomap_2": ("isomap", calculate_isomap(2, 1, dissimilarities)),
                      "k_medoids": (
                      "k_medoids", calculate_k_medoids(dissimilarities, dissimilarities_test, initial_medoids)),
                      "binary_partition": (
                      "binary_partition", calculate_binary_partition(2, dissimilarities, dissimilarities_test))
                      }
    real_value = [">50K", "<=50K"]
    dict_method = {"PCoA": [2], "neighbour": [2], "isomap": [2], "k_medoids": [real_value],
                   "binary_partition": [real_value]}

    calculate_error(data_set, data_point_y, real_value, dict_method)

    return

    # data_test_sets: dict = {"PCoA": ("PCoA", calculate_PCoA(dissimilarities_test)),
    #                         "neighbour": (
    #                             "neighbour",
    #                             evaluate_k_neighbour(2, data_point_y_test.to_numpy(), dissimilarities_test)),
    #
    #                         "isomap_2": ("isomap", calculate_isomap(2, 1, dissimilarities_test)),
    #                         "k_medoids": ("k_medoids", calculate_k_medoids(dissimilarities_test, initial_medoids)),
    #                         "binary_partition": ("binary_partition", calculate_binary_partition(2, dissimilarities_test))
    #                         }
    #

    # #
    # plot_result(data_test_sets, colors)

# def return_nothing(data_set, data_point_y):
#     return []
#
#
# def find_cut(data_set, data_point_y):
#     return
#
#

#
#
# def caculate_parameters(data_set, data_point_y):
#     return
