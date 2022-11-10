from pyclustering.cluster.kmedoids import kmedoids


def calculate_k_medoids(dissimilarities,initial_medoids):
    kmedoids_instance = kmedoids(dissimilarities, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    return kmedoids_instance.predict(dissimilarities)
