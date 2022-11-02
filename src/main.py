import os
import re
from pathlib import Path

import pandas as pd
from keras.datasets import mnist

# how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python
# how to import adult https://towardsdatascience.com/a-beginners-guide-to-data-analysis-machine-learning-with-python-adult-salary-dataset-e5fc028b6f0a
# download adult  https://archive.ics.uci.edu/ml/datasets/adult

if __name__ == '__main__':
    cwd = Path(os.getcwd())
    adult_data_path = cwd.joinpath('data/adult.data')
    adult_name_path = cwd.joinpath('data/adult.names')

    # inspire by https://stackoverflow.com/a/71480665/16619757
    with open(adult_name_path) as fp:
        cols = [sre.group('colname') for line in fp
                if (sre := re.match(r'(?P<colname>[a-z\-]+):.*\.', line))]
        cols.append('label')

    options = {'header': None, 'names': cols, 'skipinitialspace': True}

    adult_data = pd.read_csv(adult_data_path, **options)

    (data_to_train_X, data_to_train_y), (data_to_test_X, data_to_test_y) = mnist.load_data()
