import numpy as np


def calculate_euclidean_distance(datas_set: np.ndarray):
    size_of_data = datas_set.shape[0]
    dissimilarities_matrix = [[0] * size_of_data for _ in range(size_of_data)]

    for number_column in range(size_of_data):
        for number_column2 in range(number_column, size_of_data):
            dissimilarities = np.linalg.norm(datas_set[number_column] - datas_set[number_column2])
            dissimilarities_matrix[number_column][number_column2] = dissimilarities
            dissimilarities_matrix[number_column2][number_column] = dissimilarities

    return dissimilarities_matrix
