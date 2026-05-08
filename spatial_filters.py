import cv2
import numpy as np

# Mean Filter
def mean_filter(image):

    return cv2.blur(image, (5,5))


# Weighted Mean Filter
def weighted_mean_filter(image):

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ]) / 16

    return cv2.filter2D(image, -1, kernel)


# Median Filter
def median_filter(image):

    return cv2.medianBlur(image, 5)


# Max Filter
def max_filter(image):

    kernel = np.ones((3,3), np.uint8)

    return cv2.dilate(image, kernel)


# Min Filter
def min_filter(image):

    kernel = np.ones((3,3), np.uint8)

    return cv2.erode(image, kernel)


# Sharpening Filter
def sharpening_filter(image):

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return cv2.filter2D(image, -1, kernel)


# Gaussian Filter
def gaussian_filter(image):

    return cv2.GaussianBlur(image, (5,5), 0)