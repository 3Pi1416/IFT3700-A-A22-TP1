import numpy as np


def caculate_homebrew(datas_set, data_point_x_to_learn, data_point_y_to_learn):
    dict_sum = {}
    for i in range(len(data_point_x_to_learn)):
        if data_point_y_to_learn[i] in dict_sum.keys():
            dict_sum[data_point_y_to_learn[i]] = (
                dict_sum[data_point_y_to_learn[i]][0] + data_point_x_to_learn[i].astype(int),
                dict_sum[data_point_y_to_learn[i]][1] + 1)
        else:
            dict_sum[data_point_y_to_learn[i]] = (data_point_x_to_learn[i].astype(int), 1)

        dict_average = {}

    for i in dict_sum.keys():
        dict_average[i] = dict_sum[i][0] / dict_sum[i][1]

    list_of_list_of_distance = []
    size_of_data = len(datas_set)
    for i in range(size_of_data):
        list_of_distance_from_average = [0] * 10
        for j in dict_average.keys():
            list_of_distance_from_average[j] = np.linalg.norm(datas_set[i] - dict_average[j])
        list_of_distance_from_average = list_of_distance_from_average / max(list_of_distance_from_average)
        list_of_list_of_distance.append(list_of_distance_from_average)

    dissimilarities_matrix = [[0] * size_of_data for _ in range(size_of_data)]
    for number_column in range(size_of_data):
        for number_column2 in range(number_column, size_of_data):
            dissimilarities = np.linalg.norm(list_of_list_of_distance[number_column] - list_of_list_of_distance[number_column2])
            dissimilarities_matrix[number_column][number_column2] = dissimilarities
            dissimilarities_matrix[number_column2][number_column] = dissimilarities

    return dissimilarities_matrix
