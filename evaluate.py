# evaluate.py

import argparse
import os
import glob

from search.searcher import Searcher
from evaluation.ranking_evaluator import evaluate_ranking

def load_descriptor(name):
    if name == "lbp":
        from descriptors.lbp import LBPDescriptor
        return LBPDescriptor()
    elif name == "color":
        from descriptors.color_histogram import ColorHistogramDescriptor
        return ColorHistogramDescriptor()
    elif name == "resnet":
        from descriptors.resnet import ResNet18Descriptor
        return ResNet18Descriptor()
    else:
        raise ValueError(f"Descriptor '{name}' no soportado.")

def load_queries(dataset_dir):
    """Carga imágenes y etiquetas desde dataset organizado en carpetas."""
    query_paths = []
    query_labels = []
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in glob.glob(os.path.join(label_path, "*")):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                query_paths.append(fname)
                query_labels.append(label)
    return query_paths, query_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación por ranking de relevancia.")
    parser.add_argument("--dataset", type=str, required=True, help="Ruta al dataset organizado por carpetas.")
    parser.add_argument("--descriptor", type=str, required=True, choices=["lbp", "color", "resnet"], help="Descriptor usado.")
    parser.add_argument("--index", type=str, required=True, help="Ruta del índice generado.")
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Métrica de distancia.")

    args = parser.parse_args()

    descriptor = load_descriptor(args.descriptor)
    searcher = Searcher(descriptor, distance_type=args.distance)
    searcher.load_index(args.index)

    print("📥 Cargando consultas...")
    query_paths, query_labels = load_queries(args.dataset)

    print(f"🔍 Evaluando con {len(query_paths)} imágenes...")
    avg_rank = evaluate_ranking(searcher, query_paths, query_labels)

    if avg_rank is not None:
        print(f"📊 Promedio de Rank: {avg_rank:.2f}")
    else:
        print("⚠️ No se pudo calcular el ranking. ¿Dataset vacío o sin imágenes relevantes?")

