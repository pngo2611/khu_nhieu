import numpy as np

def detect_noise(image):

    gray = image.mean(axis=2)

    variance = np.var(gray)

    if variance > 5000:
        return "Gaussian Noise"

    salt = np.sum(gray == 255)
    pepper = np.sum(gray == 0)

    if salt + pepper > 500:
        return "Salt & Pepper Noise"

    return "Unknown Noise"