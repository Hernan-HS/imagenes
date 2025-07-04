# descriptors/lbp.py

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBPDescriptor:
    def __init__(self, num_points=24, radius=8, method="uniform"):
        self.num_points = num_points
        self.radius = radius
        self.method = method

    def describe(self, image):
        """
        Extrae un vector de características LBP de la imagen en escala de grises.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lbp = local_binary_pattern(image, self.num_points, self.radius, self.method)
        # Calcular histograma
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 3),
                                 range=(0, self.num_points + 2))

        # Normalizar histograma
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

