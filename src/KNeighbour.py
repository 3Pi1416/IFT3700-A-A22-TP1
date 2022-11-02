from sklearn.neighbors import KNeighborsClassifier


def evaluate_k_neighbour(data_set_x, data_set_y, test_set, dissimilarities, similarities):
    knn = KNeighborsClassifier(n_neighbors=2, metric='precomputed', algorithm='brute')
    knn.fit(dissimilarities, data_set_y)
    wow = knn.predict(test_set)
