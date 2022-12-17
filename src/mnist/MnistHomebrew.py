import pandas as pd
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from src.Analyse import analyse_Similarity
from src.mnist.SimilaritiesHomebrew import caculate_homebrew

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python


if __name__ == '__main__':
    (data_point_x, data_point_y), (data_point_x_test, data_point_y_test) = mnist.load_data()
    # size_of_data_point = 5000  #k
    size_of_data_point = len(data_point_x)

    dissimilarities_clean = caculate_homebrew(data_point_x[:size_of_data_point], data_point_x_test, data_point_y_test)

    data_point_y_clean = data_point_y[:size_of_data_point]
    test_size = 0.8

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

    colors_dict = {0: "indigo", 1: "black", 2: "red", 3: "chocolate", 4: "gold", 5: "chartreuse", 6: "turquoise",
                   7: "royalblue", 8: "orange", 9: "silver"}

    colors = [colors_dict[point] for point in data_point_y_test[:size_of_data_point]]
    initial_medoids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    real_value_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # delete list in in other list the data (from mnist)
    data_point_y = pd.DataFrame(data_point_y.tolist(), columns=[0])[0]
    data_point_y_test = pd.DataFrame(data_point_y_test.tolist(), columns=[0])[0]
    analyse_Similarity(dissimilarities_square, data_point_y, dissimilarities_test_size_of_fit, data_point_y_test,
                       initial_medoids, colors, real_value_name)
