import cv2
import numpy as np

class ColorHistogramDescriptor:
    def __init__(self, bins=(8, 8, 8)):
        self.bins = bins

    def describe(self, image):
        # Si la imagen es escala de grises, conviértela a BGR
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convertir a espacio de color HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calcular histograma y normalizar
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins,
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        return hist.flatten()

