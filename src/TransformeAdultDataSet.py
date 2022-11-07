import os
import re
from pathlib import Path

import pandas as pd
import numpy as np


def transformeAdultDataSet():
    # Transforme les données pour les prochaines étapes.
    # Les string et truple sont toujours transformer en boolean
    # Les nombre sont seulement transformer pour adult_transform_data_true_false selon la moyenne.
    adult_data: pd.DataFrame = readAdultData()

    adult_transform_data = pd.DataFrame()
    adult_transform_data_true_false = pd.DataFrame()

    for colunm in adult_data.columns:
        if np.issubdtype(adult_data[colunm].dtype, np.number):
            adult_transform_data[colunm] = adult_data[colunm].values
            adult_transform_data[colunm] = adult_data[colunm] > adult_data[colunm].mean()
        else:
            if adult_data[colunm].dtype == str or adult_data[colunm].dtype == tuple:
                for column_name in adult_data[colunm].unique():
                    adult_transform_data[column_name] = adult_data[colunm] == column_name
                    adult_transform_data_true_false[column_name] = adult_data[colunm] == column_name
            else:
                raise "bad data"

    return adult_transform_data, adult_transform_data_true_false


# how to import adult https://towardsdatascience.com/a-beginners-guide-to-data-analysis-machine-learning-with-python-adult-salary-dataset-e5fc028b6f0a
# download adult  https://archive.ics.uci.edu/ml/datasets/adult
def readAdultData():
    cwd = Path(os.getcwd())
    adult_data_path = cwd.joinpath('data/adult.data')
    adult_name_path = cwd.joinpath('data/adult.names')

    # inspire by https://stackoverflow.com/a/71480665/16619757
    with open(adult_name_path) as fp:
        cols = [sre.group('colname') for line in fp
                if (sre := re.match(r'(?P<colname>[a-z\-]+):.*\.', line))]
        cols.append('label')

    options = {'header': None, 'names': cols, 'skipinitialspace': True}

    return pd.read_csv(adult_data_path, **options)


if __name__ == '__main__':
    transformeAdultDataSet()
