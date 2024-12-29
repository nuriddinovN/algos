import numpy as np
def find_closest_centroids(X,centroids):
    """
    Find the closest centroids for each data point in X.
    Args:
        X: ndarray, shape (m, n) - Input data.
        centroids: ndarray, shape (k, n) - Centroids.
    Returns:
        idx: ndarray, shape (m,) - Index of the closest centroid for each data point.
    """
    m = len(X)
    k = len(centroids)
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        min_distance = np.inf
        for j in range(k):
            distance = np.linalg.norm(X[i] - centroids[j])
            if distance < min_distance:
                min_distance = distance
                idx[i] = j

    return idx




calc=find_closest_centroids([1,3,4,5,6,7,8,9,10],[1,2])
print(calc)