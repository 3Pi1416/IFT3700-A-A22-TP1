import pandas as pd


def calculate_similarities_equalities_matrice(data: pd.DataFrame, function_map: map):
    size_of_data = data.shape[0]
    similarities_matrix = [[0] * size_of_data for _ in range(size_of_data)]

    for number_column in range(size_of_data):
        for number_column_as_row in range(number_column, size_of_data):
            similarities = calculate_similarities_equalities(data.iloc[number_column],
                                                             data.iloc[number_column_as_row], function_map)
            similarities_matrix[number_column][number_column_as_row] = similarities
            similarities_matrix[number_column_as_row][number_column] = similarities
    return similarities_matrix, [[1 - y for y in x] for x in similarities_matrix]


def calculate_similarities_equalities(array1, array2, function_map: map) -> float:
    size_of_array = len(array1)
    if size_of_array != len(array2):
        return None

    similarities = 0
    for column in array1.keys():
        (function_in_map, args) = function_map[column]
        similarities += function_in_map(array1[column], array2[column], args)

    return similarities / size_of_array
