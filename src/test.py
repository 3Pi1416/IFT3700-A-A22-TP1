import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cosine_similarity(x, y, *args, **kwargs):
    return (x * y).sum(*args, **kwargs) / np.sqrt((x * x).sum(*args, **kwargs) * (y * y).sum(*args, **kwargs))


def cosine_dissimilarity(x, y, *args, **kwargs):
    return 1 - cosine_similarity(x, y, *args, **kwargs)


def get_cosine_dissimilarity_matrix(X, Y=None):
    Y = X if Y is None else Y
    return cosine_dissimilarity(X[:, None], Y[None, :], axis=2)


def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)


theta = np.linspace(0, 2 * np.pi, 124)[:-1]
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
circle_cosine_dissimilarity = get_cosine_dissimilarity_matrix(circle)
test = -.5 * circle_cosine_dissimilarity ** 2


agglomerative_clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
agglomerative_clustering.fit(circle_cosine_dissimilarity)

agglo_circle = agglomerative_clustering_predict(agglomerative_clustering, circle_cosine_dissimilarity)

print("wow")
