import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, denoised):
    return cv2.PSNR(original, denoised)

def calculate_ssim(original, denoised):
    score = ssim(
        original,
        denoised,
        channel_axis=2
    )
    return score