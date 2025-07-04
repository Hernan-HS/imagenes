import cv2
import numpy as np

def load_image(path, to_gray=True):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

