import numpy as np

def cosine_similarity(a, b):
    """
    Tính cosine similarity giữa hai vector đặc trưng a, b.
    Giá trị trong khoảng [-1, 1], càng gần 1 càng giống.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
