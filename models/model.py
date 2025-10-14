import tensorflow as tf
from huggingface_hub import hf_hub_download
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt

loaded_model = tf.keras.models.load_model('models/my_autoencoder.keras')

# Verify it works
loaded_model.summary()

def compute_encoded_image(image):
    image_np = np.array(image)

    # 2. Convert to float32 and normalize to [-1, 1] range
    image_np = image_np.astype(np.float32) / 127.5 - 1.0

    # 3. Add a batch dimension
    image_batch = np.expand_dims(image_np, axis=0)

    # 4. Get the model's prediction
    prediction = loaded_model.predict(image_batch)

    # 5. Remove the batch dimension and denormalize back to [0, 255] for display
    output_image = (prediction.squeeze() + 1.0) * 127.5
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)


    return output_image


