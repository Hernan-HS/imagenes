# evaluation/ranking_evaluator.py

import numpy as np
import os
import cv2

from tqdm import tqdm

def evaluate_ranking(searcher, query_paths, query_labels, top_k=100):
    ranks = []

    for i in tqdm(range(len(query_paths)), desc="📊 Evaluando"):
        path = query_paths[i]
        label = query_labels[i]

        query_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if searcher.descriptor.__class__.__name__.lower() == "lbpdescriptor" else cv2.IMREAD_COLOR)
        if query_img is None:
            continue

        results = searcher.query(query_img, top_k=top_k)

        # Posiciones de imágenes relevantes (misma clase)
        relevant_ranks = []
        for rank_idx, (res_path, res_label, _) in enumerate(results, start=1):
            if res_label == label:
                relevant_ranks.append(rank_idx)

        if relevant_ranks:
            rank_score = np.mean(relevant_ranks)
            ranks.append(rank_score)

    return np.mean(ranks) if ranks else float("inf")

