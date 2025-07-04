# indexer/feature_indexer.py

import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class FeatureIndexer:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.features = []
        self.labels = []
        self.paths = []

    def _process_image(self, img_path_label):
        img_path, label = img_path_label

        # Leer imagen a color (BGR)
        image = cv2.imread(img_path)
        if image is None:
            return None  # Imagen ilegible

        # Extraer características usando el descriptor
        feature = self.descriptor.describe(image)
        return (feature, label, img_path)

    def index_directory(self, base_path, flat_structure=False):
        if not os.path.exists(base_path):
            print(f"❌ Ruta no encontrada: {base_path}")
            return

        image_paths = []

        if flat_structure:
            image_paths = [
                (os.path.join(base_path, fname), "unknown")
                for fname in os.listdir(base_path)
                if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
        else:
            for label in os.listdir(base_path):
                label_path = os.path.join(base_path, label)
                if not os.path.isdir(label_path):
                    continue
                for fname in os.listdir(label_path):
                    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_paths.append((os.path.join(label_path, fname), label))

        if not image_paths:
            print("⚠️ No se encontraron imágenes.")
            return

        # Detectar si es ResNet para evitar multiprocessing
        descriptor_name = self.descriptor.__class__.__name__.lower()
        use_parallel = not ("resnet" in descriptor_name)

        print(f"🔍 Indexando {len(image_paths)} imágenes usando {'multiprocesamiento' if use_parallel else '1 núcleo'}...")

        if use_parallel:
            with Pool(processes=cpu_count()) as pool:
                for result in tqdm(pool.imap_unordered(self._process_image, image_paths), total=len(image_paths), desc="📸 Procesando"):
                    if result is None:
                        continue
                    feature, label, path = result
                    self.features.append(feature)
                    self.labels.append(label)
                    self.paths.append(path)
        else:
            for img_path_label in tqdm(image_paths, total=len(image_paths), desc="📸 Procesando"):
                result = self._process_image(img_path_label)
                if result is None:
                    continue
                feature, label, path = result
                self.features.append(feature)
                self.labels.append(label)
                self.paths.append(path)

        print(f"✅ Se indexaron {len(self.features)} imágenes.")

    def save_index(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path,
                            features=np.array(self.features),
                            labels=np.array(self.labels),
                            paths=np.array(self.paths))

