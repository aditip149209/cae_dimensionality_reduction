# import streamlit as st
# from PIL import Image, ImageFilter
# import numpy as np
# import matplotlib.pyplot as plt

# #Load external CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# local_css("style.css")

# #Load footer HTML
# def load_html(file_name):
#     with open(file_name) as f:
#         st.markdown(f.read(), unsafe_allow_html=True)

# #Page config
# st.set_page_config(page_title="Convolutional Autoencoder", layout="wide")

# #Title
# st.markdown("<h1>Convolutional Autoencoder</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; color:#555;'>Upload an image to see original, resized input, model output, and latent vector!</p>", unsafe_allow_html=True)
# st.markdown("---")

# #Sidebar controls
# st.sidebar.header("Settings")
# latent_size = st.sidebar.slider("Latent Vector Size", 16, 512, 128, step=16)
# simulate_blur = st.sidebar.checkbox("Simulate Model Output", value=True)

# # File uploader
# uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# # Columns for placeholders
# col1, col2, col3 = st.columns(3)

# # Create placeholders
# original_placeholder = col1.empty()
# resized_placeholder = col2.empty()
# output_placeholder = col3.empty()

# # Show placeholders before uploading
# original_placeholder.image("https://via.placeholder.com/256?text=Original+Image", caption="Original Image")
# resized_placeholder.image("https://via.placeholder.com/256?text=Resized+Input", caption="Resized Input (256x256)")
# output_placeholder.image("https://via.placeholder.com/256?text=Model+Output", caption="Model Output")

# st.markdown("---")

# # Latent vector placeholder
# latent_placeholder = st.empty()
# fig, ax = plt.subplots(figsize=(12, 3))
# ax.set_title("Latent Vector")
# latent_placeholder.pyplot(fig)

# # Update placeholders when image is uploaded
# if uploaded_file is not None:
#     # Original and resized
#     original = Image.open(uploaded_file).convert("RGB")
#     resized = original.resize((256, 256))

#     # Model output
#     if simulate_blur:
#         model_output = resized.filter(ImageFilter.BLUR)
#     else:
#         model_output = resized.copy()

#     # Update placeholders
#     original_placeholder.image(original, use_container_width=True, caption="Original Image")
#     resized_placeholder.image(resized, use_container_width=True, caption="Resized Input (256x256)")
#     output_placeholder.image(model_output, use_container_width=True, caption="Model Output")

#     # Latent vector
#     latent_vector = np.random.randn(latent_size)
#     fig, ax = plt.subplots(figsize=(12, 3))
#     ax.plot(latent_vector, color="#4B0082")
#     ax.set_title("Latent Vector", fontsize=14)
#     ax.set_ylabel("Value")
#     ax.set_xlabel("Dimension")
#     latent_placeholder.pyplot(fig)

#     # Footer
#     load_html("footer.html")

import streamlit as st
from PIL import Image, ImageFilter

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load footer HTML
def load_html(file_name):
    with open(file_name) as f:
        st.markdown(f.read(), unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Convolutional Autoencoder", layout="wide")

# Title
st.markdown("<h1>Convolutional Autoencoder</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#555;'>Upload an image to see original, resized input, and model output!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar controls
st.sidebar.header("Settings")
simulate_blur = st.sidebar.checkbox("Simulate Model Output", value=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    original = Image.open(uploaded_file).convert("RGB")
    resized = original.resize((256, 256))

    # Model output
    if simulate_blur:
        model_output = resized.filter(ImageFilter.BLUR)
    else:
        model_output = resized.copy()

    # Display images in columns
    col1, col2, col3 = st.columns(3)
    col1.image(original, use_container_width=True, caption="Original Image")
    col2.image(resized, use_container_width=True, caption="Resized Input (256x256)")
    col3.image(model_output, use_container_width=True, caption="Model Output")
else:
    st.info("Please upload an image to see previews.")

# Footer
load_html("footer.html")
