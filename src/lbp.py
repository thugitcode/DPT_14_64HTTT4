import numpy as np

def lbp_uniform(img):
    h, w = img.shape
    result = np.zeros((h-2, w-2), dtype=np.uint8)
    # offsets cho 8 pixel xung quanh
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = img[i, j]
            code = 0
            for k, (di, dj) in enumerate(offsets):
                code |= (1 if img[i+di, j+dj] >= center else 0) << k
            result[i-1, j-1] = code
    return result

def lbp_hist(img, bins=256, normalize=True):
    lbp_img = lbp_uniform(img)
    hist, _ = np.histogram(lbp_img.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    if normalize:
        s = hist.sum()
        if s > 0:
            hist /= s
    return hist
