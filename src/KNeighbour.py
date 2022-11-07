from sklearn.neighbors import KNeighborsClassifier


def evaluate_k_neighbour(data_set_y, data_set_dissimilarities, test_set_dissimilarities):
    knn = KNeighborsClassifier(n_neighbors=2, metric='precomputed', algorithm='brute')
    knn.fit(data_set_dissimilarities, data_set_y)
    wow = knn.predict(test_set_dissimilarities)

    return wow
