# from keras.datasets import mnist
import csv
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Analyse import analyse_Similarity
from src.adult.SimilaritiesEqualities import calculate_similarities_equalities_matrice
from src.adult.TransformeAdultDataSet import transformeAdultDataSet

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    adult_transform_data, function_map = transformeAdultDataSet()

    size_of_data = adult_transform_data.shape[0]
    # chose the total size of all the data
    size_of_data_point = 5000

    data_point: pd.DataFrame = adult_transform_data.iloc[
                               :size_of_data_point]  # [size_of_data_point:2*size_of_data_point])
    data_point_y_clean: pd.DataFrame = data_point["label"]
    data_point_x: pd.DataFrame = data_point.drop("label", axis=1)

    time_start = time.perf_counter()
    similarities, dissimilarities = calculate_similarities_equalities_matrice(data_point_x, function_map)
    total_time = time.perf_counter() - time_start
    print(f"Elapsed time:{total_time}")

    # saving data so it can be use later
    file_name = "similarities.csv"
    with open(file_name, 'w', newline="") as my_open_file:
        wr = csv.writer(my_open_file)
        wr.writerows(similarities)

    file_name = "dissimilarities.csv"
    with open(file_name, 'w', newline="") as my_open_file:
        wr = csv.writer(my_open_file)
        wr.writerows(dissimilarities)

    # dissimilarities, dissimilarities_test, data_point_y, data_point_y_test = train_test_split(dissimilarities,
    #                                                                                           data_point_y_clean,
    #                                                                                           test_size=0.8)
    # size = len(dissimilarities)
    #
    # dissimilarities_square = []
    # # take only the part of the
    # for line in dissimilarities:
    #     newLine = []
    #     for i in range(size):
    #         newLine.append(line[i])
    #     dissimilarities_square.append(newLine)
    #
    # dissimilarities_test_size_of_fit = []
    # for line in dissimilarities_test:
    #     newLine = []
    #     for i in range(size):
    #         newLine.append(line[i])
    #     dissimilarities_test_size_of_fit.append(newLine)
    #
    # colors = ["indigo" if point == ">50K" else "chartreuse" for point in data_point_y_test]
    # initial_medoids = [0, 1]
    #
    # analyse_Similarity(dissimilarities_square, data_point_y, dissimilarities_test_size_of_fit,
    #                    data_point_y_test, initial_medoids, colors)
