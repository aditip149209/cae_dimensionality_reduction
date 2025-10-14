import streamlit as st
from PIL import Image, ImageFilter
from models.model import compute_encoded_image
import tensorflow as tf
import numpy as np

# Load external CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Please make sure it's in the same directory.")

# Load footer HTML
def load_html(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Optional: hide footer if file not found

local_css("style.css")

# Page config
st.set_page_config(page_title="SOFTWARE ENGINEERING (IT303) COURSE PROJECT TITLE: RGB DIMENSIONALITY REDUCTION USING CONVOLUTION AUTOENCODER", layout="wide")

# Title
st.markdown("<h1>SOFTWARE ENGINEERING (IT303) COURSE PROJECT <br> TITLE: RGB DIMENSIONALITY REDUCTION USING CONVOLUTION AUTOENCODER</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#ffffff; font-size: 1.5rem'>Aditi Pandey - 231IT003<br> Prathyanga S - 231IT054</p>",
    unsafe_allow_html=True    
)
st.markdown("---")

# Sidebar controls
st.sidebar.header("Settings")


# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    original = Image.open(uploaded_file).convert("RGB")
    resized = original.resize((64, 64))
    
    # Get the model output (which is a NumPy array)
    output_array = compute_encoded_image(resized)
    
    # --- THIS IS THE NECESSARY STEP ---
    # Convert the NumPy array output back to a PIL Image so we can use it
    output_img = Image.fromarray(output_array)
    # ----------------------------------

    # Model output processing (now works correctly)
    model_output = output_img.copy()

    # Display images in columns
    st.subheader("Image Comparison")
    col1, col2, col3 = st.columns(3)
    col1.image(original, use_container_width=True, caption="Original Image")
    col2.image(resized, use_container_width=True, caption="Resized Input (64x64)") # Corrected caption
    col3.image(model_output, use_container_width=True, caption="Model Output")
else:
    st.info("Please upload an image to see previews.")

# Footer
load_html("footer.html")
