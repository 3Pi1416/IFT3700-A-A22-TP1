import pandas as pd


def calculate_sorensen_dice_matrice(data: pd.DataFrame):
    size_of_data = data.shape[0]
    similarities_matrix = [[0] * size_of_data] * size_of_data

    # au cas ou les données son fragmenter
    data_ = data.copy()
    for number_column in range(size_of_data):
        for number_column_as_row in range(number_column, size_of_data):
            similarities = calculate_sorensen_dice(data_.iloc[number_column], data_.iloc[number_column_as_row])
            similarities_matrix[number_column][number_column_as_row] = similarities
            similarities_matrix[number_column_as_row][number_column] = similarities
    return similarities_matrix, [[1 - y for y in x] for x in similarities_matrix]


def calculate_sorensen_dice(array1, array2):
    size_of_array = len(array1)
    if size_of_array != len(array2):
        return None

    similarities = 0
    for i in range(size_of_array):
        similarities += int(array1[i] == array2[i])

    return similarities / size_of_array
