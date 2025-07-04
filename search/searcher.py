# search/searcher.py

import numpy as np
import cv2
from utils import distance_metrics

class Searcher:
    def __init__(self, descriptor, distance_type='euclidean'):
        self.descriptor = descriptor
        self.distance_func = self._get_distance_func(distance_type)
        self.index = None

    def _get_distance_func(self, distance_type):
        if distance_type == 'euclidean':
            return distance_metrics.euclidean_distance
        elif distance_type == 'cosine':
            return distance_metrics.cosine_distance
        else:
            raise ValueError(f"Tipo de distancia no soportado: {distance_type}")

    def load_index(self, index_path):
        data = np.load(index_path, allow_pickle=True)
        self.features = data['features']
        self.labels = data['labels']
        self.paths = data['paths']

    def query(self, query_image, top_k=10):
        query_vector = self.descriptor.describe(query_image)
        results = []

        for feat, label, path in zip(self.features, self.labels, self.paths):
            d = self.distance_func(query_vector, feat)
            results.append((path, label, d))

        results.sort(key=lambda x: x[2])
        return results[:top_k]

