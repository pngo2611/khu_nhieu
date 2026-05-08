import cv2
import numpy as np

def fft_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    return fshift


# Low-pass Filter
def low_pass_filter(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape

    crow, ccol = rows//2 , cols//2

    mask = np.zeros((rows, cols), np.uint8)

    r = 30

    center = [crow, ccol]

    x, y = np.ogrid[:rows, :cols]

    mask_area = (
        (x - center[0])**2 +
        (y - center[1])**2
    ) <= r*r

    mask[mask_area] = 1

    fshift = fshift * mask

    ishift = np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(ishift)

    img_back = np.abs(img_back)

    return np.uint8(img_back)


# High-pass Filter
def high_pass_filter(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape

    crow, ccol = rows//2 , cols//2

    mask = np.ones((rows, cols), np.uint8)

    r = 30

    x, y = np.ogrid[:rows, :cols]

    mask_area = (
        (x - crow)**2 +
        (y - ccol)**2
    ) <= r*r

    mask[mask_area] = 0

    fshift = fshift * mask

    ishift = np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(ishift)

    img_back = np.abs(img_back)

    return np.uint8(img_back)


# High-Boost Filter
def high_boost_filter(image, k=1.5):

    blurred = cv2.GaussianBlur(image, (5,5), 0)

    mask = image - blurred

    boosted = image + k * mask

    return np.clip(boosted, 0, 255).astype(np.uint8)


# Laplace Filter
def laplace_filter(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return np.uint8(np.absolute(lap))


# Butterworth Filter
def butterworth_lowpass_filter(image, D0=30, n=2):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rows, cols = gray.shape

    crow, ccol = rows//2, cols//2

    u = np.arange(rows)

    v = np.arange(cols)

    U, V = np.meshgrid(u, v, indexing='ij')

    D = np.sqrt((U-crow)**2 + (V-ccol)**2)

    H = 1 / (1 + (D / D0)**(2*n))

    F = np.fft.fft2(gray)

    Fshift = np.fft.fftshift(F)

    G = H * Fshift

    Gshift = np.fft.ifftshift(G)

    img_back = np.fft.ifft2(Gshift)

    img_back = np.abs(img_back)

    return np.uint8(img_back)