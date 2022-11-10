import numpy as np
from matplotlib import pyplot


def plot_result(data_sets: dict, colors):
    for data_name, values in data_sets.items():
        data_type = values[0]
        data_set = values[1]

        fig = pyplot.figure()
        ax = fig.add_subplot()
        ax.set_title(data_name)

        if data_type == 2:
            ax.scatter(data_set, np.zeros_like(data_set), c=colors)
        else:
            ax.scatter(range(len(data_set)), data_set, c=colors)

        pyplot.show()
