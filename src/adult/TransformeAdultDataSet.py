import os
import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.adult.CompareFunctions import compare_string, compare_number


def transformeAdultDataSet():
    # Transforme les données pour les prochaines étapes.
    # Les string et truple sont toujours transformer en boolean
    # Les nombre sont seulement transformer pour adult_transform_data_true_false selon la moyenne.
    adult_data: pd.DataFrame = readAdultData()
    function_map_for_Columns: map(str, tuple) = {}

    for column in adult_data.columns:
        if np.issubdtype(adult_data[column].dtype, np.number):
            function_map_for_Columns[column] = (compare_number, (adult_data[column].min(), adult_data[column].max()))
        else:
            function_map_for_Columns[column] = (compare_string, None)

    return adult_data, function_map_for_Columns


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
