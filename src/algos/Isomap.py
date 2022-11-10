from sklearn.manifold import Isomap


def calculate_isomap(number_neighbours, number_components, dissimilarities):
    isomap = Isomap(n_components=number_components, n_neighbors=number_neighbours, metric='precomputed')
    return isomap.fit_transform(dissimilarities)
