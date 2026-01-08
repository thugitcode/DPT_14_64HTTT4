import numpy as np
import cv2

def gabor_kernels(
    ksize=31,
    sigmas=(4.0, 8.0),
    lambdas=(8.0, 16.0),
    gammas=(0.5,),
    thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)
):
    """
    Sinh ra nhiều kernel Gabor với các tham số khác nhau
    - ksize: kích thước kernel
    - sigma: độ rộng Gaussian
    - lambda: bước sóng
    - gamma: tỉ lệ aspect
    - theta: góc (hướng)
    """
    kernels = []
    for sigma in sigmas:
        for lambd in lambdas:
            for gamma in gammas:
                for theta in thetas:
                    k = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lambd, gamma, 0,
                        ktype=cv2.CV_32F
                    )
                    kernels.append(k)
    return kernels

def gabor_features(img, kernels=None):
    """
    Tính đặc trưng Gabor bằng cách lọc ảnh với nhiều kernel
    Sau đó lấy mean và std của đáp ứng làm đặc trưng
    """
    if kernels is None:
        kernels = gabor_kernels()
    feats = []
    for k in kernels:
        resp = cv2.filter2D(img, cv2.CV_32F, k)
        feats.append(resp.mean())
        feats.append(resp.std())
    return np.array(feats, dtype=np.float32)
