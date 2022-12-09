import numpy as np
from sklearn.decomposition import KernelPCA


def calculate_PCoA(dissimilarities,dissimilarities_test):
    # comme vue en class
    pcoa = KernelPCA(n_components=1, kernel='precomputed')
    return pcoa.fit_transform(-.5 * np.array(dissimilarities) ** 2)
