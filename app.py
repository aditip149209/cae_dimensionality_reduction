import io
import requests
import base64
import streamlit as st
from models.model import compute_encoded_image
from authlib.integrations.requests_client import OAuth2Session
from PIL import Image
import mysql.connector
from db.db_config import get_db_connection
from db.db_funcs import get_image_history, get_image_history_list, add_image_to_history, create_user
from auth.auth import get_authorization_url, fetch_user_info, fetch_token
import os

PAGE_TITLE = (
    "RGB Dimensionality Reduction using Convolutional Autoencoder"
)
PAGE_SUBTITLE = "SOFTWARE ENGINEERING (IT303) COURSE PROJECT"
AUTHORS = "Aditi Pandey - 231IT003<br>Prathyanga S - 231IT054"

# Auth0 Configuration - Fetched from Streamlit Secrets
try:
    CLIENT_ID = (os.environ.get("AUTH0_CLIENT_ID") or st.secrets["AUTH0_CLIENT_ID"]).strip('"').strip("'")
    CLIENT_SECRET = (os.environ.get("AUTH0_CLIENT_SECRET") or st.secrets["AUTH0_CLIENT_SECRET"]).strip('"').strip("'")
    DOMAIN = (os.environ.get("AUTH0_DOMAIN") or st.secrets["AUTH0_DOMAIN"]).strip('"').strip("'")
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

# --- HISTORY CALLBACKS (The "Asynchronous" Part) ---
def load_history_image(image_id):
    """
    Callback function. Fetches a specific image from the DB
    and stores it in the session state.
    """
    record = get_image_history(image_id)
    st.session_state.selected_history_item = record

def show_upload_page():
    """Callback to clear the history view and show the uploader."""
    st.session_state.selected_history_item = None


# --- UI COMPONENTS ---
def show_login_page():
    """Displays the login page."""
    st.markdown("<h1>Welcome to the Autoencoder Project</h1>", unsafe_allow_html=True)
    st.write("Please log in to upload an image and use the application.")
    auth_url = get_authorization_url()
    st.link_button("Login with Auth0", auth_url, use_container_width=True)


def show_main_application(user):
    """Displays the main image processing application."""
    user_id = user['sub']

    create_user(st.session_state.user_info)
    user = st.session_state.user_info

    # Sidebar for user info and logout
    st.sidebar.header(f"Welcome, {user.get('name', user.get('email'))}!")
    st.sidebar.image(user.get('picture'), width=100)
    if st.sidebar.button("Logout"):
        st.session_state.user_info = None
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.subheader("Your Upload History")

    

    #we will write the logic for the history here. this will require fetching the result of db queries and storing them in an array and looping over them 
    history_items = get_image_history_list(user_id)
    
    if not history_items:
        st.sidebar.caption("No uploads found.")
    else:
        # Create a button for each history item
        for item in history_items:
            date_str = item['created_at'].strftime("%Y-%m-%d %H:%M")
            item_label = f"{item['original_filename']} (on {date_str})"
            
            # --- This is the "React" part ---
            # When clicked, the 'on_click' callback fires,
            # fetches the data, stores it in state, and reruns.
            st.sidebar.button(
                item_label, 
                key=f"history_{item['id']}", 
                on_click=load_history_image, 
                args=(item['id'],) # Pass the image ID to the callback
            )
    
    # Main page content
    st.markdown(f"<h1>{PAGE_SUBTITLE}<br>{PAGE_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; color:#ffffff; font-size: 1.5rem'>{AUTHORS}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "selected_history_item" in st.session_state and st.session_state.selected_history_item:
        # --- VIEW HISTORY ITEM PAGE ---
        st.subheader("Viewing from Your History")
        
        record = st.session_state.selected_history_item
        st.markdown(f"**Original Filename:** `{record['original_filename']}`")
        image_bytes = base64.b64decode(record['reconstructed_image_data'])
        
        # The 'reconstructed_image_data' is the Base64 data URL
        st.image(image_bytes, caption="Reconstructed Image")
        
        if st.button("Upload a New Image"):
            # This callback clears the state and reruns
            show_upload_page()
            st.rerun()
            
    else:
        # --- UPLOAD PAGE (Default) ---
        uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")
        
        if uploaded_file:
            process_and_display_image(uploaded_file, user_id)
        else:
            st.info("Please upload an image to see previews.")
    load_html("footer.html")



def process_and_display_image(uploaded_file, user_id):
    original_filename = uploaded_file.name

    original = Image.open(uploaded_file).convert("RGB")
    resized = original.resize((64, 64))

    output_array = compute_encoded_image(resized)

    model_output = Image.fromarray(output_array)

    buffered = io.BytesIO()
    model_output.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    encoded_img = base64.b64encode(image_bytes).decode('utf-8')

    add_image_to_history(user_id, original_filename, encoded_img)


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
    st.session_state.history_needs_refresh = True
    st.session_state.uploader_key += 1




def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    local_css("style.css")

    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

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
        show_main_application(st.session_state.user_info)
    else:
        show_login_page()


if __name__ == "__main__":
    main()
