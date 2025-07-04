# searcher/similarity_searcher.py

import numpy as np
from scipy.spatial import distance

class SimilaritySearcher:
    def __init__(self, index):
        self.index = index

    def search(self, query_vector, metric='chi2'):
        results = {}

        for filename, vector in self.index.items():
            if metric == 'chi2':
                dist = self._chi2_distance(query_vector, vector)
            elif metric == 'cosine':
                dist = distance.cosine(query_vector, vector)
            else:
                raise ValueError("Métrica no soportada: usa 'chi2' o 'cosine'")
            results[filename] = dist

        # Ordenar por menor distancia
        results = sorted(results.items(), key=lambda x: x[1])
        return results

    def _chi2_distance(self, histA, histB, eps=1e-10):
        return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

