import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_linear_similarities(data_x: pd.DataFrame, data_y: pd.DataFrame):
    linear_regressor = LinearRegression()
    size_y = data_y.shape[0]
    predict_y = [[0] * data_y.shape[0]] * size_y
    for column in data_y.columns:
        linear_regressor.fit(data_x.iloc[:], column)
        predict_y.append(linear_regressor.predict(data_x.iloc[:]))

    return predict_y
