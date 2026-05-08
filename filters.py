import cv2

def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def median_filter(image):
    return cv2.medianBlur(image, 5)

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def nlm_filter(image):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )