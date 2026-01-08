import cv2

def load_image(path):
    # đọc ảnh grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img

def preprocess(img, size=(128, 128)):
    # resize ảnh về kích thước chuẩn
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img
