# query.py

import argparse
import os
import cv2
from search.searcher import Searcher
from utils.distance_metrics import cosine_distance, euclidean_distance
from utils.visualization import show_query_results  # 👈 Importamos función de visualización

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
    parser = argparse.ArgumentParser(description="Buscar imagen similar.")
    parser.add_argument("--query", type=str, required=True, help="Ruta de la imagen de consulta.")
    parser.add_argument("--descriptor", type=str, default="lbp", choices=["lbp", "color", "resnet"], help="Tipo de descriptor.")
    parser.add_argument("--index", type=str, required=True, help="Ruta del archivo de índice.")
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Métrica de distancia.")
    parser.add_argument("--topk", type=int, default=5, help="Número de resultados a mostrar.")
    parser.add_argument("--output", type=str, default=None, help="Ruta para guardar la visualización (opcional).")  # 👈 nuevo argumento

    args = parser.parse_args()
    descriptor = load_descriptor(args.descriptor)

    if not os.path.exists(args.index):
        print(f"❌ Índice no encontrado en {args.index}.")
        exit()

    query_image = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE if args.descriptor == "lbp" else cv2.IMREAD_COLOR)
    if query_image is None:
        print("❌ No se pudo leer la imagen de consulta.")
        exit()

    searcher = Searcher(descriptor, distance_type=args.distance)
    searcher.load_index(args.index)

    results = searcher.query(query_image, top_k=args.topk)

    print("✅ Resultados más similares:")
    for i, (path, label, score) in enumerate(results):
        print(f"[{i+1}] {label} - {os.path.basename(path)} (Distancia: {score:.4f})")

    # 👇 Mostrar los resultados visualmente
    show_query_results(args.query, results, output_path=args.output)

