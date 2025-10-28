# RGB Image Dimensionality Reduction using a Convolutional Autoencoder

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Hub-yellow.svg)

This repository demonstrates dimensionality reduction on RGB images with a Convolutional Autoencoder (CAE).
It includes a training notebook (`models/cae.ipynb`), a Streamlit demo (`app.py`) that uses the trained model, helper utilities, and the saved model artifacts in `models/`.

This README explains how to set up the environment, run the Streamlit app, reproduce training/evaluation from the notebook, and where to find important files.

## Quick overview

- Streamlit demo: `app.py` (interactive image upload + model output)
- Model logic: `models/model.py` (loads `models/my_autoencoder.keras` and exposes `compute_encoded_image`)
- Training & experiments: `models/cae.ipynb` (Jupyter notebook with dataset pipeline, model definition, training, evaluation, visualizations)
- Helpers: `processor/image_utils.py` (image save/preprocess utilities)
- Saved model files: `models/my_autoencoder.keras`, `models/my_autoencoder.weights.h5`

## Repository structure

```
cae_dimensionality_reduction/
├─ app.py                      # Streamlit UI
├─ main.py                     # (empty helper / placeholder)
├─ requirements.txt            # Python dependencies
├─ style.css                   # Streamlit UI styling
├─ footer.html                 # Optional footer for Streamlit app
├─ models/
│  ├─ model.py                 # loads model and provides compute_encoded_image()
│  ├─ cae.ipynb                # training and evaluation notebook
│  ├─ my_autoencoder.keras     # saved Keras model (used by app)
│  └─ my_autoencoder.weights.h5
├─ processor/
│  └─ image_utils.py           # saving + preprocessing helpers
├─ public/
│  └─ uploads/                 # saved uploads (created at runtime)
└─ README.md
```

## Prerequisites

- Linux / macOS / Windows with WSL
- Python 3.8+ (tested on 3.10 in the included virtualenv)
- ~8–16 GB RAM (training requires more; inference is lightweight)

Recommended: create and use a virtual environment.

## Setup (local)

1. Create & activate a virtual environment

```bash
# Create venv (if not already present)
python3 -m venv cae_env
# Activate (Linux/macOS)
source cae_env/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes the packages used by the Streamlit demo and notebook (TensorFlow, Streamlit, numpy, matplotlib, scikit-image, huggingface_hub).
- If you use a GPU-enabled TensorFlow, install the matching `tensorflow` package version with GPU support.

3. Configure secrets and environment

    The application requires a few secrets and environment variables for authentication and the database. There are two supported ways to provide them:

    - Preferred (container / CI friendly): provide them as environment variables (recommended for Docker and production).
    - Local development: create a `.streamlit/secrets.toml` file with the values.

    Example (recommended) — set environment variables locally or via `docker-compose`:

    ```bash
    export AUTH0_CLIENT_ID="<your-auth0-client-id>"
    export AUTH0_CLIENT_SECRET="<your-auth0-client-secret>"
    export AUTH0_DOMAIN="<your-auth0-domain>"            # e.g. dev-xxxxxx.us.auth0.com

    export DB_HOST=localhost
    export DB_USER=aditi
    export DB_PASS=root
    export DB_NAME=cae_lol
    ```

    Example (local `.streamlit/secrets.toml`) — create `.streamlit/secrets.toml` in the project root:

    ```toml
    # .streamlit/secrets.toml
    AUTH0_CLIENT_ID = "your-client-id"
    AUTH0_CLIENT_SECRET = "your-client-secret"
    AUTH0_DOMAIN = "dev-xxxxxx.us.auth0.com"

    DB_HOST = "host.docker.internal"
    DB_USER = "aditi"
    DB_PASS = "root"
    DB_NAME = "cae_lol"
    ```

    Important notes:
    - Do NOT check `secrets.toml` into version control. It contains sensitive credentials.
    - The app code prefers environment variables first, then falls back to Streamlit secrets. This makes running in Docker or CI more predictable.

    Auth0 configuration (quick guide):
    1. Go to https://manage.auth0.com and create a new Application (Regular Web Application).
    2. In the Application settings set the following:
         - Allowed Callback URLs: `http://localhost:8501`
         - Allowed Logout URLs: `http://localhost:8501`
         - Allowed Web Origins: `http://localhost:8501`
    3. Copy the **Client ID**, **Client Secret**, and **Domain** into your environment variables or `.streamlit/secrets.toml`.

    Database initialization:
    - The repository contains `init.sql` which creates the required schema (users, user_images). If you use the included `docker-compose.yml` the SQL will be executed automatically by the MySQL container on first startup.
    - If you run MySQL yourself, run:

       ```bash
       mysql -u root -p < init.sql
       ```

    Data stored by the app:
    - Users are stored in the `users` table (`uid`, `username`). `uid` is the Auth0 `sub` value.
    - Uploaded / reconstructed images are stored in `user_images` with columns: `id`, `uid`, `original_filename`, `reconstructed_image_data` (Base64), `created_at`.

    The app saves reconstructed images (PNG) as base64 text in the database; the history sidebar fetches the list from `user_images` and displays past uploads for the logged-in user.

## Run with Docker (recommended)

This project includes a `docker-compose.yml` that runs both the Streamlit app and a MySQL database. The `docker-compose.yml` is already configured with sensible defaults (database name `cae_lol`, user `aditi`, password `root`).

1. Build and start containers:

```bash
docker compose up --build
```

2. The Streamlit app will be available at `http://localhost:8501`.

Notes about secrets and Docker:
- For security you should NOT embed production secrets into `docker-compose.yml`. Instead, provide them via a host `.env` file or your CI/CD secret store. Example `docker-compose` environment snippet:

```yaml
services:
   app:
      environment:
         - AUTH0_CLIENT_ID=${AUTH0_CLIENT_ID}
         - AUTH0_CLIENT_SECRET=${AUTH0_CLIENT_SECRET}
         - AUTH0_DOMAIN=${AUTH0_DOMAIN}
         - DB_HOST=db
         - DB_USER=aditi
         - DB_PASS=root
         - DB_NAME=cae_lol
```

- If you prefer using a `.streamlit/secrets.toml`, ensure it is mounted into the container at `/app/.streamlit/secrets.toml` (or copy it into the image). Example docker-compose volume mount:

```yaml
      volumes:
         - ./ .streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro
```

Troubleshooting common secrets/auth issues:
- Streamlit will raise a `StreamlitSecretNotFoundError` if no secrets file is present and you attempt to access `st.secrets[...]`. The app prefers environment variables first; if you rely on `.streamlit/secrets.toml` make sure the file exists in the container.
- If the Auth0 domain resolves with percent-encoded quotes (e.g. contains `%22`), that means extra quotes were embedded in the value. Use raw values (no surrounding extra quotes) and prefer environment variables. You can debug the value by printing it with `!r` in Python to reveal stray quotes.


## Run the Streamlit app (demo)

The simplest way to see the model working is to run the Streamlit UI.

```bash
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501). Upload a JPG/PNG image. The app will:

- resize the uploaded image to 64×64
- pass it to `models.compute_encoded_image()` which uses the saved model
- display original, resized input and model output side-by-side

If you get an error saying the model file is missing (`models/my_autoencoder.keras`), either:

- run the training notebook to produce `my_autoencoder.keras` (see Training section), or
- download a pre-trained model and place it in `models/` as `my_autoencoder.keras`.

## Use the model programmatically

The function `compute_encoded_image(image: PIL.Image)` is provided in `models/model.py`. It expects a PIL Image (RGB) and returns a NumPy uint8 image (H, W, 3) ready for display or saving.

Example (Python):

```python
from PIL import Image
from models.model import compute_encoded_image

img = Image.open('some_photo.jpg').convert('RGB').resize((64, 64))
output = compute_encoded_image(img)  # NumPy uint8 array in [0..255]
Image.fromarray(output).save('recon.png')
```

## Training & evaluation (Jupyter Notebook)

The training pipeline and experiments are in `models/cae.ipynb`.

Highlights and how the notebook is organized:

- Loads `zh-plus/tiny-imagenet` from Hugging Face datasets and sets up an efficient tf.data pipeline.
- Defines a compact convolutional autoencoder for 64×64 images (`build_compact_autoencoder`).
- Compiles and trains the model (MSE loss). Training hyperparameters are defined in the notebook (LEARNING_RATE, BATCH_SIZE, IMG_SIZE, NUM_EPOCHS).
- Contains helper cells for visualization (reconstructions, activations) and evaluation metrics (MSE, PSNR, SSIM).
- Saves the trained model to `models/my_autoencoder.keras` and weights to `models/my_autoencoder.weights.h5`.

How to run the notebook:

1. Start Jupyter (or JupyterLab):

```bash
jupyter lab  # or jupyter notebook
```

2. Open `models/cae.ipynb` and run the cells in order. The notebook already contains preprocessing steps, model definition, training, evaluation and visualization.

Notes about dataset: the notebook uses the Hugging Face dataset `zh-plus/tiny-imagenet`. Make sure you have network access and space for the dataset.

## Evaluation & metrics

Within the notebook there are example cells that:

- compute MSE via `model.evaluate(valid_dataset)`
- compute PSNR and SSIM on a validation batch using TensorFlow's `tf.image.psnr` and `tf.image.ssim`
- visualize reconstructions and intermediate layer activations

Example output (from the notebook):

- Mean Squared Error (MSE) on validation set: printed via `model.evaluate`
- Average PSNR on batch: printed with `tf.reduce_mean(psnr_value)`
- Average SSIM on batch: printed with `tf.reduce_mean(ssim_value)`

## Files of interest

- `app.py` — Streamlit demo that imports `compute_encoded_image` and shows original/resized/model output
- `models/model.py` — Loads the saved Keras model (`models/my_autoencoder.keras`) at import time and exposes `compute_encoded_image()`
- `models/cae.ipynb` — Notebook with full training, evaluation and visualization workflow






