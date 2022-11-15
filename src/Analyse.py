from src.algos.BinaryPartitioning import calculate_binary_partition
from src.algos.Isomap import calculate_isomap
from src.algos.KMedoids import calculate_k_medoids
from src.algos.KNeighbour import evaluate_k_neighbour
from src.algos.PCoA import calculate_PCoA
from src.caculateError import calculate_error
from src.plotResult import plot_result


def analyse_Similarity(dissimilarities, similarities, data_point_y, dissimilarities_test, similarities_test,
                       data_point_y_test, initial_medoids, colors):
    data_set: dict = {"PCoA": ("PCoA", calculate_PCoA(dissimilarities)),
                      "neighbour_2": ("neighbour", evaluate_k_neighbour(2, data_point_y.to_numpy(), similarities)),
                      "neighbour_3": ("neighbour", evaluate_k_neighbour(3, data_point_y.to_numpy(), similarities)),
                      "neighbour_4": ("neighbour", evaluate_k_neighbour(4, data_point_y.to_numpy(), similarities)),
                      "neighbour_5": ("neighbour", evaluate_k_neighbour(5, data_point_y.to_numpy(), similarities)),
                      "isomap_2": ("isomap", calculate_isomap(2, 1, dissimilarities)),
                      "k_medoids": ("k_medoids", calculate_k_medoids(dissimilarities, initial_medoids)),
                      "binary_partition": (2, calculate_binary_partition(2, dissimilarities))
                      }

    data_test_sets: dict = {"PCoA": ("PCoA", calculate_PCoA(dissimilarities_test)),
                            "neighbour_2": (
                                "neighbour", evaluate_k_neighbour(2, data_point_y_test.to_numpy(), similarities_test)),
                            "neighbour_3": (
                                "neighbour", evaluate_k_neighbour(3, data_point_y_test.to_numpy(), similarities_test)),
                            "neighbour_4": (
                                "neighbour", evaluate_k_neighbour(4, data_point_y_test.to_numpy(), similarities_test)),
                            "neighbour_5": (
                                "neighbour", evaluate_k_neighbour(5, data_point_y_test.to_numpy(), similarities_test)),
                            "isomap_2": ("isomap", calculate_isomap(2, 1, dissimilarities_test)),
                            "k_medoids": ("k_medoids", calculate_k_medoids(dissimilarities_test, initial_medoids)),
                            "binary_partition": (2, calculate_binary_partition(2, dissimilarities))
                            }

    calculate_error(data_set, data_test_sets, data_point_y, [">50K", "<=50K"])
    plot_result(data_test_sets, colors)


def return_nothing(data_set, data_point_y):
    return []


def find_cut(data_set, data_point_y):
    return


dict_method = {"PCoA": find_cut, "neighbour": 3, "isomap": find_cut, "k_medoids": 3, "binary_partition": 3}


def caculate_parameters(data_set, data_point_y):
    return