import numpy as np


def cosine_similarity(x, y, *args, **kwargs):
    return (x * y).sum(*args, **kwargs) / np.sqrt((x * x).sum(*args, **kwargs) * (y * y).sum(*args, **kwargs))


def cosine_dissimilarity(x, y, *args, **kwargs):
    return 1 - cosine_similarity(x, y, *args, **kwargs)


def get_cosine_dissimilarity_matrix(X, Y=None):
    Y = X if Y is None else Y
    return cosine_dissimilarity(X[:, None], Y[None, :], axis=2)


theta = np.linspace(0, 2 * np.pi, 124)[:-1]
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
circle_cosine_dissimilarity = get_cosine_dissimilarity_matrix(circle)
test = -.5*circle_cosine_dissimilarity**2

print("wow")
