import numpy as np
from sklearn.cluster import AgglomerativeClustering


def calculate_binary_partition(number_clusters, dissimilarities):
    agglomerative_clustering = AgglomerativeClustering(n_clusters=number_clusters, affinity='precomputed',
                                                       linkage='average')
    agglomerative_clustering.fit(dissimilarities)
    return agglomerative_clustering_predict(agglomerative_clustering, dissimilarities)


def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    # comme vue en classe
    average_dissimilarity = list()
    dissimilarity_matrix_temp = np.array(dissimilarity_matrix)
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix_temp[:, np.where(agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)
