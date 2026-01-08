import numpy as np
from .lbp import lbp_hist
from .gabor import gabor_features

def extract_features(img):
    # Lấy vector LBP (256 chiều)
    lbp_vec = lbp_hist(img)

    # Lấy vector Gabor (ví dụ 32 chiều)
    gabor_vec = gabor_features(img)

    # Ghép lại thành một vector duy nhất
    feat = np.concatenate([lbp_vec, gabor_vec], axis=0)

    # Chuẩn hóa L2 để tránh ảnh hưởng bởi độ lớn
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    return feat
