# from keras.datasets import mnist
import pandas as pd

from src.Analyse import analyse_Similarity
from src.adult.SimilaritiesEqualities import calculate_similarities_equalities_matrice
from src.adult.TransformeAdultDataSet import transformeAdultDataSet

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    adult_transform_data, function_map = transformeAdultDataSet()

    size_of_data = adult_transform_data.shape[0]
    size_of_data_point = 100  # int(size_of_data * 0.8)

    data_point: pd.DataFrame = adult_transform_data.iloc[
                               :size_of_data_point]  # [size_of_data_point:2*size_of_data_point])
    data_point_y: pd.DataFrame = data_point["label"]
    data_point_x: pd.DataFrame = data_point.drop("label", axis=1)

    size_of_test = size_of_data_point  # size_of_data - size_of_data_point
    data_point_test: pd.DataFrame = adult_transform_data.iloc[
                                    size_of_data_point:size_of_data_point + size_of_test]
    data_point_y_test: pd.DataFrame = data_point_test["label"]
    data_point_x_test: pd.DataFrame = data_point_test.drop("label", axis=1)

    similarities, dissimilarities = calculate_similarities_equalities_matrice(data_point_x, function_map)
    similarities_test, dissimilarities_test = calculate_similarities_equalities_matrice(data_point_x_test, function_map)

    colors = ["indigo" if point == ">50K" else "chartreuse" for point in data_point_y_test]
    initial_medoids = [0, 1]

    analyse_Similarity(dissimilarities, similarities, data_point_y, dissimilarities_test, similarities_test,
                       data_point_y_test, initial_medoids, colors)
