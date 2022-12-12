# from keras.datasets import mnist
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Analyse import analyse_Similarity
from src.adult.SimilaritiesEqualities import calculate_similarities_equalities_matrice
from src.adult.TransformeAdultDataSet import transformeAdultDataSet


# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python

def read_data_and_save(size_of_data_point):
    adult_transform_data, function_map = transformeAdultDataSet()

    # chose the total size of all the data

    data_point: pd.DataFrame = adult_transform_data.iloc[:size_of_data_point]
    data_point_x: pd.DataFrame = data_point.drop("label", axis=1)

    similarities, dissimilarities = calculate_similarities_equalities_matrice(data_point_x, function_map)

    # saving data so it can be use later
    file_name = "similarities.csv"
    with open(file_name, 'w', newline="") as my_open_file:
        csv_writer = csv.writer(my_open_file)
        csv_writer.writerows(similarities)

    file_name = "dissimilarities.csv"
    with open(file_name, 'w', newline="") as my_open_file:
        csv_writer = csv.writer(my_open_file)
        csv_writer.writerows(dissimilarities)


def add_y_and_prepare_data(dissimilarities_clean, test_size):
    # add y to the matrix
    adult_transform_data, function_map = transformeAdultDataSet()
    data_point: pd.DataFrame = adult_transform_data.iloc[:size_of_data_point]
    data_point_y_clean: pd.DataFrame = data_point["label"]

    # shuffle a false, pour avoir la matrice diagonal 0 lorsqu'on apprend les modèles.
    dissimilarities, dissimilarities_test, data_point_y, data_point_y_test = train_test_split(dissimilarities_clean,
                                                                                              data_point_y_clean,
                                                                                              test_size=test_size,
                                                                                              shuffle=False)
    size = len(dissimilarities)

    dissimilarities_square = []
    # take only the part of the
    for line in dissimilarities:
        newLine = []
        for i in range(size):
            newLine.append(line[i])
        dissimilarities_square.append(newLine)

    dissimilarities_test_size_of_fit = []
    for line in dissimilarities_test:
        newLine = []
        for i in range(size):
            newLine.append(line[i])
        dissimilarities_test_size_of_fit.append(newLine)

    return dissimilarities_square, dissimilarities_test_size_of_fit, data_point_y, data_point_y_test


if __name__ == '__main__':
    cwd = Path(os.getcwd())

    dissimilarities_files = cwd.joinpath("save", 'dissimilarities_5000.csv')
    # similarities_files = cwd.joinpath("save", 'similarities_5000.csv')
    size_of_data_point = 5000
    if not dissimilarities_files.exists():
        read_data_and_save(size_of_data_point)

    # dissimilarities = pd.read_csv(interest_text_file)

    dissimilarities_clean = []
    with open(dissimilarities_files, "r") as my_open_file:
        csv_reader = csv.reader(my_open_file)
        for row in csv_reader:
            dissimilarities_clean.append([float(i) for i in row])

    dissimilarities_square, dissimilarities_test_size_of_fit, data_point_y, data_point_y_test = add_y_and_prepare_data(
        dissimilarities_clean, 0.8)

    colors = ["indigo" if point == ">50K" else "chartreuse" for point in data_point_y_test]
    initial_medoids = [0, 1]
    real_value_name = [">50K", "<=50K"]

    analyse_Similarity(dissimilarities_square, data_point_y, dissimilarities_test_size_of_fit, data_point_y_test,
                       initial_medoids, colors, real_value_name)
