import numpy as np
import cv2

# 1. White Noise
def add_white_noise(image):

    noise = np.random.randint(
        0,
        256,
        image.shape,
        dtype=np.uint8
    )

    noisy = cv2.addWeighted(
        image,
        0.8,
        noise,
        0.2,
        0
    )

    return noisy


# 2. Gaussian Noise
def add_gaussian_noise(image):

    mean = 0
    sigma = 25

    gauss = np.random.normal(
        mean,
        sigma,
        image.shape
    )

    noisy = image + gauss

    return np.clip(noisy, 0, 255).astype(np.uint8)


# 3. Uniform / Quantization Noise
def add_uniform_noise(image):

    noise = np.random.uniform(
        -30,
        30,
        image.shape
    )

    noisy = image + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


# 4. Poisson Noise
def add_poisson_noise(image):

    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))

    noisy = np.random.poisson(
        image * vals
    ) / float(vals)

    return np.clip(noisy, 0, 255).astype(np.uint8)


# 5. Impulse Noise
def add_impulse_noise(image):

    noisy = image.copy()

    prob = 0.02

    rnd = np.random.rand(*image.shape[:2])

    noisy[rnd < prob] = 255

    return noisy


# 6. Salt & Pepper Noise
def add_salt_pepper_noise(image):

    noisy = image.copy()

    prob = 0.02

    rnd = np.random.rand(*image.shape[:2])

    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255

    return noisy