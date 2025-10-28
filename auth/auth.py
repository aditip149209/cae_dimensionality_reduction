import streamlit as st
import io
import requests
from authlib.integrations.requests_client import OAuth2Session
import os


# --- Auth0 Configuration (Dual-Mode) ---
CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID") or st.secrets.get("AUTH0_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET") or st.secrets.get("AUTH0_CLIENT_SECRET")
DOMAIN = os.environ.get("AUTH0_DOMAIN") or st.secrets.get("AUTH0_DOMAIN")

# This also needs to be flexible.
# For local/docker, it's localhost. For AWS, it will be your public URL.
REDIRECT_URI = os.environ.get("REDIRECT_URI") or "http://localhost:8501"

# --- Validation Check ---
# Stops the app if secrets are missing from *both* sources
if not all([CLIENT_ID, CLIENT_SECRET, DOMAIN, REDIRECT_URI]):
    st.error(
        "Auth0 secrets not found. Please add them to .streamlit/secrets.toml or environment variables."
    )
    st.stop()

# --- Endpoints (calculated from the config) ---
AUTHORIZATION_ENDPOINT = f"https://{DOMAIN}/authorize"
TOKEN_ENDPOINT = f"https://{DOMAIN}/oauth/token"
USERINFO_ENDPOINT = f"https://{DOMAIN}/userinfo"
LOGOUT_ENDPOINT = f"https://{DOMAIN}/v2/logout"


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