import io
from contextlib import contextmanager
import mysql.connector
import requests
import streamlit as st
from models.model import compute_encoded_image
from authlib.integrations.requests_client import OAuth2Session
from PIL import Image



PAGE_TITLE = (
    "RGB Dimensionality Reduction using Convolutional Autoencoder"
)
PAGE_SUBTITLE = "SOFTWARE ENGINEERING (IT303) COURSE PROJECT"
AUTHORS = "Aditi Pandey - 231IT003<br>Prathyanga S - 231IT054"

# Auth0 Configuration - Fetched from Streamlit Secrets
try:
    CLIENT_ID = st.secrets["AUTH0_CLIENT_ID"]
    CLIENT_SECRET = st.secrets["AUTH0_CLIENT_SECRET"]
    DOMAIN = st.secrets["AUTH0_DOMAIN"]
    REDIRECT_URI = "http://localhost:8501"
except KeyError:
    st.error(
        "Auth0 secrets not found. Please add them to .streamlit/secrets.toml"
    )
    st.stop()

AUTHORIZATION_ENDPOINT = f"https://{DOMAIN}/authorize"
TOKEN_ENDPOINT = f"https://{DOMAIN}/oauth/token"
USERINFO_ENDPOINT = f"https://{DOMAIN}/userinfo"
LOGOUT_ENDPOINT = f"https://{DOMAIN}/v2/logout"


# --- HELPER FUNCTIONS ---
def local_css(file_name: str):
    """Loads a local CSS file into the Streamlit app."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found.")


def load_html(file_name: str):
    """Loads a local HTML file into the Streamlit app."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Silently fail if footer is not found


@contextmanager
def get_db_connection():
    """Manages the MySQL connection, ensuring it's properly closed."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            database=st.secrets["DB_NAME"],
        )
        yield conn
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        yield None
    finally:
        if "conn" in locals() and conn.is_connected():
            conn.close()


# --- AUTHENTICATION LOGIC ---
def get_oauth_session():
    """Initializes and returns an OAuth2 session."""
    return OAuth2Session(CLIENT_ID, CLIENT_SECRET, redirect_uri=REDIRECT_URI)


def get_authorization_url() -> str:
    """Generates the Auth0 login URL."""
    session = get_oauth_session()
    scope = "openid profile email"
    audience = f"https://{DOMAIN}/userinfo"
    auth_url, state = session.create_authorization_url(
        AUTHORIZATION_ENDPOINT, audience=audience, scope=scope, prompt='login'
    )
    st.session_state.oauth_state = state
    return auth_url


def fetch_token(code: str) -> dict or None:
    """Exchanges the authorization code for an access token."""
    session = get_oauth_session()
    try:
        return session.fetch_token(TOKEN_ENDPOINT, code=code)
    except Exception as e:
        st.error(f"Error fetching token: {e}")
        return None


def fetch_user_info(token: dict) -> dict or None:
    """Uses the access token to get user information."""
    session = get_oauth_session()
    session.token = token
    try:
        response = session.get(USERINFO_ENDPOINT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching user info: {e}")
        return None


# --- UI COMPONENTS ---
def show_login_page():
    """Displays the login page."""
    st.markdown("<h1>Welcome to the Autoencoder Project</h1>", unsafe_allow_html=True)
    st.write("Please log in to upload an image and use the application.")
    auth_url = get_authorization_url()
    st.link_button("Login with Auth0", auth_url, use_container_width=True)


def show_main_application():
    """Displays the main image processing application."""
    user = st.session_state.user_info

    # Sidebar for user info and logout
    st.sidebar.header(f"Welcome, {user.get('name', user.get('email'))}!")
    st.sidebar.image(user.get('picture'), width=100)
    if st.sidebar.button("Logout"):
        st.session_state.user_info = None
        st.rerun()

    # Main page content
    st.markdown(f"<h1>{PAGE_SUBTITLE}<br>{PAGE_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; color:#ffffff; font-size: 1.5rem'>{AUTHORS}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        process_and_display_image(uploaded_file)
    else:
        st.info("Please upload an image to see the results.")

    load_html("footer.html")


def process_and_display_image(uploaded_file):
    """Handles the image processing and displays the results."""
    original = Image.open(uploaded_file).convert("RGB")
    resized = original.resize((64, 64))
    output_array = compute_encoded_image(resized)
    model_output = Image.fromarray(output_array)

    st.subheader("Image Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)
    with col2:
        st.image(resized, caption="Resized Input (64x64)", use_container_width=True)
    with col3:
        st.image(model_output, caption="Model Output", use_container_width=True)

        # Convert PIL image to a byte stream for download
        buf = io.BytesIO()
        model_output.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Output Image",
            data=byte_im,
            file_name="output_image.png",
            mime="image/png",
        )


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    local_css("style.css")

    if "user_info" not in st.session_state:
        st.session_state.user_info = None

    query_params = st.query_params
    if "code" in query_params and st.session_state.user_info is None:
        code = query_params["code"]
        token = fetch_token(code)
        if token:
            user_info = fetch_user_info(token)
            if user_info:
                st.session_state.user_info = user_info
                st.query_params.clear()
                st.rerun()

    if st.session_state.user_info:
        show_main_application()
    else:
        show_login_page()


if __name__ == "__main__":
    main()
