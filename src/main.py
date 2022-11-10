# from keras.datasets import mnist
import pandas as pd

from src.TransformeAdultDataSet import transformeAdultDataSet
from src.algos.BinaryPartitioning import calculate_binary_partition
from src.algos.Isomap import calculate_isomap
from src.algos.KMedoids import calculate_k_medoids
from src.algos.KNeighbour import evaluate_k_neighbour
from src.algos.PCoA import calculate_PCoA
from src.plotResult import plot_result
from src.similaritiesEqualities import calculate_similarities_equalities_matrice

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    adult_transform_data, function_map = transformeAdultDataSet()

    size_of_data = adult_transform_data.shape[0]
    size_of_data_point = 500  # int(size_of_data * 0.8)

    data_point: pd.DataFrame = adult_transform_data.iloc[:size_of_data_point]
    data_point_y: pd.DataFrame = data_point["label"]
    data_point_x: pd.DataFrame = data_point.drop("label", axis=1)

    # size_of_test = 100  # size_of_data - size_of_data_point
    # data_point_test: pd.DataFrame = adult_transform_data.iloc[
    #                                 size_of_data_point:size_of_data_point + size_of_test]
    # data_point_y_test: pd.DataFrame = data_point_test["label"]
    # data_point_x_test: pd.DataFrame = data_point_test.drop("label", axis=1)

    similarities, dissimilarities = calculate_similarities_equalities_matrice(data_point_x, function_map)
    # similarities_test, dissimilarities_test = calculate_similarities_equalities_matrice(data_point_x_test, function_map)

    colors = ["indigo" if point == ">50K" else "chartreuse" for point in data_point_y]
    data_sets: dict = {"PCoA": (2, calculate_PCoA(dissimilarities)),
                       "neighbour_2": (3, evaluate_k_neighbour(2, data_point_y.to_numpy(), dissimilarities)),
                       # "neighbour_3": (3, evaluate_k_neighbour(3, data_point_y.to_numpy(), dissimilarities)),
                       # "neighbour_4": (3, evaluate_k_neighbour(4, data_point_y.to_numpy(), dissimilarities)),
                       # "neighbour_5": (3, evaluate_k_neighbour(5, data_point_y.to_numpy(), dissimilarities)),
                       "isomap_2": (2, calculate_isomap(2, 1, dissimilarities)),
                       "k_medoids": (3, calculate_k_medoids(dissimilarities)),
                       "binary_partition": (2, calculate_binary_partition(2, dissimilarities))
                       }

    # sum_error = sum([test[i] != data_point_y_test.iloc[i] for i in range(len(test))])
    # (data_to_train_X, data_to_train_y), (data_to_test_X, data_to_test_y) = mnist.load_data()
    plot_result(data_sets, colors)

    print("wow")
