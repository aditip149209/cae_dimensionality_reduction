import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Create the folder if it doesn't exist
UPLOAD_FOLDER = 'public/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_image(uploaded_file):
    """Saves a Streamlit UploadedFile to a public folder."""
    if uploaded_file is not None:
        # Create a path to save the file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        
        # Write the file to the path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None


def preprocess_image(image_path):
    """Loads and preprocesses an image to a tensor for the model."""
    # Load the image using Keras utilities
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(256, 256)
    )
    
    # Convert the image to a NumPy array and normalize it
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    
    # Add a batch dimension (e.g., shape changes from (256, 256, 3) to (1, 256, 256, 3))
    image_tensor = np.expand_dims(image_array, axis=0)
    
    return image_tensor

