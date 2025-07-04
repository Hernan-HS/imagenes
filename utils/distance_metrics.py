# utils/distance_metrics.py

import numpy as np
from scipy.spatial import distance

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def cosine_distance(v1, v2):
    return distance.cosine(v1, v2)

