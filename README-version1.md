# RGB Image Dimensionality Reduction using a Convolutional Autoencoder

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hub-yellow.svg)

> **⚠️ Note: This is a DRAFT version.**
>
> The documentation and setup instructions will be updated to include Dockerization for easier deployment and environment management.

---

This project demonstrates how to perform dimensionality reduction on RGB images using a pre-trained Denoising Convolutional Autoencoder (CAE). The model is loaded from the Hugging Face Hub, and its components—the encoder and decoder—are isolated to perform specific tasks.

The core of the project is to take a high-dimensional input (an RGB image) and use the model's encoder to generate a low-dimensional latent vector that represents the image's most important features.

---

##  Core Concepts 💡

-   **Convolutional Autoencoder (CAE):** An unsupervised neural network that uses an encoder-decoder structure to learn a compressed representation (latent space) of its input.
-   **Encoder:** The part of the network that compresses the input image down to a low-dimensional latent vector. This vector is the result of the dimensionality reduction.
-   **Decoder:** The part of the network that reconstructs the image from the compressed latent vector.
-   **Denoising Autoencoder:** A type of autoencoder trained to reconstruct a *clean* image from a *noisy* input. This forces the encoder to learn robust and meaningful features, making it excellent for dimensionality reduction.



---

##  Features

-   Load a pre-trained Keras model from the Hugging Face Hub.
-   Isolate the **encoder** to perform dimensionality reduction.
-   Isolate the **decoder** to generate new images from latent vectors.
-   Demonstrate the model's original denoising capability.

---

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a `requirements.txt` file:**
    Create a file named `requirements.txt` and add the following dependencies:
    ```
    tensorflow
    huggingface_hub
    numpy
    matplotlib
    scikit-image
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the model**
    ```bash
    python3 model.py
    ```

---