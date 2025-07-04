# config.py

# Ruta a la base de datos de imágenes
DATASET_PATH = "data/CBIR_50"

# Ruta donde se guardarán los vectores de características
INDEX_SAVE_PATH = "indexes/lbp_index.npz"

# Tipo de descriptor que se usará
DESCRIPTOR_TYPE = "lbp"

# Métrica de distancia: 'euclidean' o 'cosine'
DISTANCE_METRIC = "cosine"

# True si las imágenes están organizadas en subcarpetas por clase
DATASET_HAS_LABELS = True

