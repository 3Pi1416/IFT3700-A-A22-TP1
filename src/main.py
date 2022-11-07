# from keras.datasets import mnist
from src.KNeighbour import evaluate_k_neighbour
from src.TransformeAdultDataSet import transformeAdultDataSet
import pandas as pd

from src.sorensen_dice import calculate_sorensen_dice_matrice

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    adult_transform_data, adult_transform_data_true_false = transformeAdultDataSet()

    size_of_data = adult_transform_data_true_false.shape[0]
    size_of_data_point = 1000  # int(size_of_data * 0.8)
    size_of_test = 500  # size_of_data - size_of_data_point

    data_point: pd.DataFrame = adult_transform_data_true_false.iloc[:size_of_data_point]
    data_point_y: pd.DataFrame = data_point["<=50K"]
    data_point_x: pd.DataFrame = data_point.drop("<=50K", axis=1).drop(">50K", axis=1)

    data_point_test: pd.DataFrame = adult_transform_data_true_false.iloc[
                                    size_of_data_point:size_of_data_point + size_of_test]
    data_point_y_test: pd.DataFrame = data_point["<=50K"]
    data_point_x_test: pd.DataFrame = data_point.drop("<=50K", axis=1).drop(">50K", axis=1)

    similarities, dissimilarities = calculate_sorensen_dice_matrice(data_point_x)
    similarities_test, dissimilarities_test = calculate_sorensen_dice_matrice(data_point_x_test)
    test = evaluate_k_neighbour(data_point_y.to_numpy(), dissimilarities, dissimilarities_test)
    # (data_to_train_X, data_to_train_y), (data_to_test_X, data_to_test_y) = mnist.load_data()

    print("wow")
