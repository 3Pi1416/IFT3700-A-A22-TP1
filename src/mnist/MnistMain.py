from keras.datasets import mnist

from src.algos.EuclideanDistance import calculate_euclidean_distance
from src.algos.Isomap import calculate_isomap
from src.algos.KMedoids import calculate_k_medoids
from src.algos.KNeighbour import evaluate_k_neighbour
from src.algos.PCoA import calculate_PCoA
from src.plotResult import plot_result

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    (data_point_x, data_point_y), (data_point_x_test, data_point_y_test) = mnist.load_data()
    # size_of_data = data_point_x.shape[0]
    size_of_data_point = 500  #

    similarities = calculate_euclidean_distance(
        data_point_x[:size_of_data_point])  # data_point_x[size_of_data_point:2*size_of_data_point])
    # similarities_test, dissimilarities_test = calculatke_similarities_equalities_matrice(data_point_x_test, function_map)

    colors_dict = {0: "indigo", 1: "black", 2: "red", 3: "chocolate", 4: "gold", 5: "chartreuse", 6: "turquoise",
                   7: "royalblue", 8: "orange", 9: "silver"}
    colors = [colors_dict[point] for point in data_point_y[:size_of_data_point]]
    initial_medoids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data_sets: dict = {"PCoA": (2, calculate_PCoA(similarities)),
                       "neighbour_2": (3, evaluate_k_neighbour(2, data_point_y[:size_of_data_point], similarities)),
                       # "neighbour_3": (3, evaluate_k_neighbour(3, data_point_y.to_numpy(), dissimilarities)),
                       # "neighbour_4": (3, evaluate_k_neighbour(4, data_point_y.to_numpy(), dissimilarities)),
                       # "neighbour_5": (3, evaluate_k_neighbour(5, data_point_y.to_numpy(), dissimilarities)),
                       "isomap_10": (3, calculate_isomap(10, 1, similarities)),
                       "k_medoids": (3, calculate_k_medoids(similarities, initial_medoids)),
                       # "binary_partition": (2, calculate_binary_partition(2, dissimilarities))
                       }

    # sum_error = sum([test[i] != data_point_y_test.iloc[i] for i in range(len(test))])

    plot_result(data_sets, colors)

    print("wow")
