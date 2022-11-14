import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

data_type = {
                "PCoA":, "neighbour":, "isomap":, "k_medoids":
}

def plot_result(data_sets: dict, colors):
    with (PdfPages('result.pdf')) as pp:
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
            pp.savefig(pyplot.gcf())
