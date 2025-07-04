import argparse
import os
import cv2
import numpy as np

from indexer.feature_indexer import FeatureIndexer
from search.searcher import Searcher
from utils.distance_metrics import cosine_distance, euclidean_distance

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


def load_distance_function(name):
    if name == "cosine":
        return cosine_distance
    elif name == "euclidean":
        return euclidean_distance
    else:
        raise ValueError(f"Distancia '{name}' no soportada.")

def main():
    parser = argparse.ArgumentParser(description="Sistema CBIR simple.")
    parser.add_argument("--mode", type=str, required=True, choices=["index", "query"],
                        help="Modo de operación: index o query.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Ruta a la carpeta del dataset.")
    parser.add_argument("--index", type=str,
                        help="Ruta para guardar o cargar el índice. Si no se especifica, se asigna automáticamente según descriptor.")
    parser.add_argument("--descriptor", type=str, default="lbp", choices=["lbp", "color", "resnet"],
                        help="Método de extracción de características.")
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"],
                        help="Tipo de distancia para comparación.")
    parser.add_argument("--query", type=str,
                        help="Ruta de la imagen de consulta (solo en modo query).")
    parser.add_argument("--flat", action="store_true",
                        help="Indica si las imágenes están en una sola carpeta (sin subcarpetas).")

    args = parser.parse_args()

    descriptor = load_descriptor(args.descriptor)

    # Asignar nombre del índice automáticamente si no se especifica
    if not args.index:
        index_name = f"{args.descriptor}_index.npz"
        args.index = os.path.join("indexes", index_name)

    if args.mode == "index":
        print(f"🔍 Indexando base de datos con descriptor '{args.descriptor}'...")
        indexer = FeatureIndexer(descriptor)
        indexer.index_directory(args.dataset, flat_structure=args.flat)
        indexer.save_index(args.index)
        print(f"✅ Índice guardado en {args.index}")

    elif args.mode == "query":
        if not args.query or not os.path.exists(args.query):
            print("❌ Debes especificar una imagen de consulta válida con --query.")
            return

        if not os.path.exists(args.index):
            print(f"❌ Índice no encontrado en {args.index}. Ejecuta primero en modo 'index'.")
            return

        print(f"🔎 Realizando búsqueda con descriptor '{args.descriptor}' y distancia '{args.distance}'...")

        # Leer imagen de consulta
        query_image = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE if args.descriptor == "lbp" else cv2.IMREAD_COLOR)
        if query_image is None:
            print("❌ No se pudo leer la imagen.")
            return

        searcher = Searcher(descriptor, distance_type=args.distance)
        searcher.load_index(args.index)

        results = searcher.query(query_image, top_k=5)

        print("✅ Resultados más similares:")
        for i, (path, label, score) in enumerate(results):
            print(f"[{i+1}] {label} - {os.path.basename(path)} (Distancia: {score:.4f})")

if __name__ == "__main__":
    main()

