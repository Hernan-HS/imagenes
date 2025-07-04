# index.py

import argparse
import os
from indexer.feature_indexer import FeatureIndexer
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar imágenes.")
    parser.add_argument("--dataset", type=str, required=True, help="Ruta al dataset.")
    parser.add_argument("--descriptor", type=str, default="lbp", choices=["lbp", "color", "resnet"], help="Tipo de descriptor.")
    parser.add_argument("--index", type=str, help="Ruta de salida del índice.")
    parser.add_argument("--flat", action="store_true", help="Indica si el dataset no tiene subcarpetas.")

    args = parser.parse_args()
    descriptor = load_descriptor(args.descriptor)

    index_path = args.index or f"indexes/{args.descriptor}_index.npz"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    print(f"🔍 Indexando con descriptor '{args.descriptor}'...")
    indexer = FeatureIndexer(descriptor)
    indexer.index_directory(args.dataset, flat_structure=args.flat)
    indexer.save_index(index_path)
    print(f"✅ Índice guardado en: {index_path}")

