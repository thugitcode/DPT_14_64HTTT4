import os
import numpy as np
from .preprocessing import load_image, preprocess
from .features import extract_features

def build_index(dataset_dir):
    """
    Duyệt toàn bộ ảnh trong dataset_dir, trích đặc trưng và lưu vào index.
    Đường dẫn trả về chuẩn Flask: bắt đầu bằng 'static/...'
    """
    index = []
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(supported_ext):
                continue
            full_path = os.path.join(root, fname)

            # tạo đường dẫn tương đối từ static/
            rel_path = os.path.relpath(full_path, start="static")
            flask_path = os.path.join("static", rel_path).replace("\\", "/")

            try:
                img = load_image(full_path)
                img = preprocess(img)
                vec = extract_features(img)
                index.append((flask_path, vec))
            except Exception as e:
                print(f"Skip {full_path}: {e}")
    return index

def cosine_similarity_matrix(qvec, index_matrix):
    qnorm = np.linalg.norm(qvec)
    inorms = np.linalg.norm(index_matrix, axis=1)
    sims = (index_matrix @ qvec) / (inorms * qnorm + 1e-10)
    return sims

def search(query_path, index, topk=5):
    qimg = load_image(query_path)
    qimg = preprocess(qimg)
    qvec = extract_features(qimg)

    paths = [path for path, vec in index]
    index_matrix = np.array([vec for _, vec in index])

    sims = cosine_similarity_matrix(qvec, index_matrix)
    top_ids = np.argsort(-sims)[:topk]

    results = [(paths[i], sims[i]) for i in top_ids]
    return results
