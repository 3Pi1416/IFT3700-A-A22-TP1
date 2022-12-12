from sklearn.neighbors import KNeighborsClassifier


def evaluate_k_neighbour(number_neighbours, data_set_y, data_set_dissimilarities, data_set_dissimilarities_test,metric='precomputed' ):
    knn = KNeighborsClassifier(n_neighbors=number_neighbours, metric=metric, algorithm='brute')
    knn.fit(data_set_dissimilarities, data_set_y)
    return knn.predict(data_set_dissimilarities_test)