import tensorflow as tf
from huggingface_hub import hf_hub_download
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt

# # loading the model. once loaded, it is stored in the cache of the host
# model_path = hf_hub_download(repo_id="BueormLLC/AID_turbo", filename="model.h5")
# autoencoder = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
# autoencoder.summary()

# # finding the bottleneck layer- analysed the model to find the layer at which dimensions start increasing 
# bottleneck_layer_name = "conv2d_7"
# encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(bottleneck_layer_name).output)

# # i printed a summary for the encoder
# encoder.summary()

# # creating decoder 
# latent_input_shape = (32, 32, 256)
# latent_input = tf.keras.layers.Input(shape=latent_input_shape, name="decoder-input")

# x = autoencoder.get_layer("up_sampling2d")(latent_input)
# x = autoencoder.get_layer("conv2d_8")(x)
# x = autoencoder.get_layer('batch_normalization_8')(x)
# x = autoencoder.get_layer('up_sampling2d_1')(x)
# x = autoencoder.get_layer('conv2d_9')(x)
# x = autoencoder.get_layer('batch_normalization_9')(x)
# x = autoencoder.get_layer('up_sampling2d_2')(x)
# x = autoencoder.get_layer('conv2d_10')(x)
# x = autoencoder.get_layer('batch_normalization_10')(x)


# decoder_output = autoencoder.get_layer("conv2d_11")(x)

# decoder = tf.keras.Model(inputs=latent_input, outputs=decoder_output, name="decoder")

# # i printed a summary for the decoder 
# decoder.summary()

# # get latent vector
# def get_latent_vector(encoder, image_tensor):
#     latent_representation = encoder.predict(image_tensor)
    
#     # Flatten the feature map into a single vector
#     latent_vector = latent_representation.flatten()
    
#     return latent_vector


# def get_reconstructed_image(decoder, latent_vector):
#     reconstructed_image = decoder.predict(latent_vector)

#     return reconstructed_image

# # # Load the image
# test_image = tf.keras.preprocessing.image.load_img(
#     'Screenshot from 2025-06-13 10-40-59.png', 
#     target_size=(256, 256)
# )

# # # Convert to a 3D array -> shape: (256, 256, 3)
# test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)

# # # === FIX IS HERE ===
# # # Add a 4th dimension for the batch -> shape: (1, 256, 256, 3)
# test_image_batch = np.expand_dims(test_image_array, axis=0)

# # # Normalize the image data (important step!)
# test_image_batch = test_image_batch / 255.0

# # print(test_image_batch.dtype)
# # Now, pass the correctly shaped and normalized batch to your function
# # res = autoencoder.predict(test_image_batch)
# # print(res)

# latent_vector = get_latent_vector(encoder, test_image_batch)
# print(latent_vector)

# nan_count = np.isnan(latent_vector).sum()
# print(f"Number of NaNs: {nan_count}")


# total_elements = latent_vector.size
# print(f"total number of elements {total_elements}")


# STEP 2: Provide the custom function to the loader
# custom_objects = {'mse': mse}

# Now, load the model with the custom objects dictionary
loaded_model = tf.keras.models.load_model('my_autoencoder.keras')

# Verify it works
loaded_model.summary()



