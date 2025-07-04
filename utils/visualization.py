# utils/visualization.py

import cv2
import matplotlib.pyplot as plt
import os

def show_query_results(query_path, results, output_path=None):
    """
    Muestra la imagen de consulta y los resultados más similares.

    Args:
        query_path (str): Ruta de la imagen de consulta.
        results (list): Lista de tuplas (path, label, score).
        output_path (str): Ruta para guardar la visualización (opcional).
    """
    num_results = len(results)
    cols = min(num_results, 5)
    rows = 1 + (num_results // 5)

    plt.figure(figsize=(3 * (cols + 1), 3 * rows))

    # Mostrar imagen de consulta
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, cols + 1, 1)
    plt.imshow(query_img)
    plt.title("Consulta")
    plt.axis("off")

    # Mostrar resultados
    for i, (img_path, label, score) in enumerate(results):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols + 1, i + 2)
        plt.imshow(img)
        plt.title(f"{label}\n{score:.2f}")
        plt.axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"🖼️ Visualización guardada en {output_path}")
    else:
        plt.show()

