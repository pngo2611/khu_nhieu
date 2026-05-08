import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from PIL import Image
from skimage.metrics import structural_similarity as ssim

from noise import *
from spatial_filters import *
from frequency_filters import *


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Image Denoising Studio",
    layout="wide"
)

st.title("🧠 Smart Image Denoising Studio")
st.markdown("### Digital Image Processing Project")


# =========================================================
# SAFE IMAGE FUNCTION
# =========================================================

def safe_image(img):

    img = np.nan_to_num(img)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return img.astype(np.uint8)


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(original, processed):

    processed = safe_image(processed)

    if len(processed.shape) == 2:

        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2GRAY
        )

    return cv2.PSNR(
        original,
        processed
    )


# =========================================================
# SSIM
# =========================================================

def calculate_ssim(original, processed):

    processed = safe_image(processed)

    if len(processed.shape) == 2:

        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2GRAY
        )

        score = ssim(
            original,
            processed
        )

    else:

        score = ssim(
            original,
            processed,
            channel_axis=2
        )

    return score


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # =====================================================
    # READ IMAGE
    # =====================================================

    image = Image.open(uploaded_file)

    image = np.array(image)

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    image = safe_image(image)

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("⚙ Settings")

    noise_type = st.sidebar.selectbox(
        "Choose Noise",
        [
            "White Noise",
            "Gaussian Noise",
            "Uniform Noise",
            "Poisson Noise",
            "Impulse Noise",
            "Salt & Pepper Noise"
        ]
    )

    filter_type = st.sidebar.selectbox(
        "Choose Filter",
        [
            # Spatial Domain
            "Mean Filter",
            "Weighted Mean Filter",
            "Median Filter",
            "Max Filter",
            "Min Filter",
            "Sharpening Filter",
            "Gaussian Filter",

            # Frequency Domain
            "Low Pass Filter",
            "High Pass Filter",
            "High Boost Filter",
            "Laplace Filter",
            "Butterworth Filter"
        ]
    )

    # =====================================================
    # ADD NOISE
    # =====================================================

    if noise_type == "White Noise":

        noisy = add_white_noise(image)

    elif noise_type == "Gaussian Noise":

        noisy = add_gaussian_noise(image)

    elif noise_type == "Uniform Noise":

        noisy = add_uniform_noise(image)

    elif noise_type == "Poisson Noise":

        noisy = add_poisson_noise(image)

    elif noise_type == "Impulse Noise":

        noisy = add_impulse_noise(image)

    else:

        noisy = add_salt_pepper_noise(image)

    noisy = safe_image(noisy)

    # =====================================================
    # APPLY FILTER
    # =====================================================

    start_time = time.time()

    # ---------------- SPATIAL ----------------

    if filter_type == "Mean Filter":

        denoised = mean_filter(noisy)

    elif filter_type == "Weighted Mean Filter":

        denoised = weighted_mean_filter(noisy)

    elif filter_type == "Median Filter":

        denoised = median_filter(noisy)

    elif filter_type == "Max Filter":

        denoised = max_filter(noisy)

    elif filter_type == "Min Filter":

        denoised = min_filter(noisy)

    elif filter_type == "Sharpening Filter":

        denoised = sharpening_filter(noisy)

    elif filter_type == "Gaussian Filter":

        denoised = gaussian_filter(noisy)

    # ---------------- FREQUENCY ----------------

    elif filter_type == "Low Pass Filter":

        denoised = low_pass_filter(noisy)

    elif filter_type == "High Pass Filter":

        denoised = high_pass_filter(noisy)

    elif filter_type == "High Boost Filter":

        denoised = high_boost_filter(noisy)

    elif filter_type == "Laplace Filter":

        denoised = laplace_filter(noisy)

    else:

        denoised = butterworth_lowpass_filter(noisy)

    end_time = time.time()

    processing_time = end_time - start_time

    denoised = safe_image(denoised)

    # =====================================================
    # METRICS
    # =====================================================

    psnr_value = calculate_psnr(
        image,
        denoised
    )

    ssim_value = calculate_ssim(
        image,
        denoised
    )

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🖼 Original")

        st.image(
            image,
            use_container_width=True,
            clamp=True
        )

    with col2:

        st.subheader("⚠ Noisy")

        st.image(
            noisy,
            use_container_width=True,
            clamp=True
        )

    with col3:

        st.subheader("✨ Processed")

        st.image(
            denoised,
            use_container_width=True,
            clamp=True
        )

    # =====================================================
    # METRICS DISPLAY
    # =====================================================

    st.markdown("---")

    st.subheader("📊 Performance Metrics")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "PSNR",
            f"{psnr_value:.2f} dB"
        )

    with m2:

        st.metric(
            "SSIM",
            f"{ssim_value:.4f}"
        )

    with m3:

        st.metric(
            "Processing Time",
            f"{processing_time:.4f} sec"
        )

    # =====================================================
    # HISTOGRAM
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Histogram Comparison")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(
        image.ravel(),
        256,
        [0,256],
        alpha=0.5,
        label="Original"
    )

    ax.hist(
        noisy.ravel(),
        256,
        [0,256],
        alpha=0.5,
        label="Noisy"
    )

    ax.hist(
        denoised.ravel(),
        256,
        [0,256],
        alpha=0.5,
        label="Processed"
    )

    ax.legend()

    st.pyplot(fig)

    # =====================================================
    # FFT VISUALIZATION
    # =====================================================

    st.markdown("---")

    st.subheader("🌌 Frequency Spectrum")

    fft = fft_image(noisy)

    magnitude = 20 * np.log(
        np.abs(fft) + 1
    )

    magnitude = safe_image(magnitude)

    st.image(
        magnitude,
        use_container_width=True,
        clamp=True
    )

    # =====================================================
    # COMPARE FILTERS
    # =====================================================

    st.markdown("---")

    st.subheader("🔥 Compare Spatial Filters")

    comparison_results = {

        "Mean":
            mean_filter(noisy),

        "Median":
            median_filter(noisy),

        "Gaussian":
            gaussian_filter(noisy),

        "Sharpen":
            sharpening_filter(noisy)
    }

    cols = st.columns(4)

    for idx, (name, img) in enumerate(comparison_results.items()):

        img = safe_image(img)

        psnr_temp = calculate_psnr(
            image,
            img
        )

        with cols[idx]:

            st.image(
                img,
                use_container_width=True,
                clamp=True
            )

            st.write(f"### {name}")

            st.write(
                f"PSNR: {psnr_temp:.2f}"
            )

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    st.markdown("---")

    download_image = safe_image(denoised)

    _, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(
            download_image,
            cv2.COLOR_RGB2BGR
        )
    )

    st.download_button(
        label="📥 Download Processed Image",
        data=buffer.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )

else:

    st.info("👆 Please upload an image to start.")