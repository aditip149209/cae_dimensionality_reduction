from contextlib import contextmanager
import mysql.connector
import streamlit as st
import io
import base64
from db.db_config import get_db_connection
from auth.auth import fetch_token, fetch_user_info, get_authorization_url, get_oauth_session


def create_user(user_info):
    """
    Finds a user by their Auth0 'sub' ID (uid) and creates or updates them.
    This is an "upsert" operation.
    """
    # Use the 'sub' field from Auth0 as our unique 'uid'
    user_id = user_info['sub']
    
    # Use the 'name' field, but fall back to 'email' if 'name' doesn't exist
    username = user_info.get('name', user_info.get('email'))
    
    # If no name or email, we can't create a user (violates NOT NULL)
    if not username:
        print(f"Error: User {user_id} has no name or email. Cannot sync to DB.")
        st.error("Could not sync user: missing name and email.")
        return

    sql_query = """
    INSERT INTO users (uid, username)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE username = %s;
    """
    
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                # We pass 'username' twice:
                # 1. For the INSERT 'username' value
                # 2. For the UPDATE 'username' value
                cursor.execute(sql_query, (user_id, username, username))
                conn.commit()  # Save the changes to the database
                cursor.close()
    except mysql.connector.Error as err:
        # Log the error for debugging
        print(f"Database Error in create_or_update_user: {err}")
        # Optionally show a non-technical error to the user
        st.warning("There was a problem syncing your user account.")


def add_image_to_history(user_id, image_url, reconstructed_image):

    sql_query = """
    INSERT INTO user_images (uid, original_filename, reconstructed_image_data, created_at)
    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
    ON DUPLICATE KEY UPDATE created_at = CURRENT_TIMESTAMP;
    """

    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, (user_id, image_url, reconstructed_image))
                conn.commit()  # Commit the transaction to save the change
                cursor.close()
    except mysql.connector.Error as err:
        print(f"Database Error in add_image_to_history: {err}")
        st.error("Error saving image to history.")

def get_image_history_list(user_id):
    sql_query = """
    SELECT id, original_filename, created_at 
    FROM user_images
    WHERE uid = %s
    ORDER BY created_at DESC 
    """

    history_items = []

    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(sql_query, (user_id, ))
                history_items = cursor.fetchall()
                cursor.close()


    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        st.error("Error fetching image history list")

    return history_items


def get_image_history(id):
    # This SQL command selects the image URL and its creation time
    # for the specific user, ordering by the newest first and limiting to 10.
    sql_query = """
    SELECT original_filename, created_at, reconstructed_image_data
    FROM user_images
    WHERE id = %s
    ORDER BY created_at DESC
    """

    history_items = []
    try:
        with get_db_connection() as conn:
            if conn:
                # dictionary=True makes the results easy to use (e.g., item['image_url'])
                cursor = conn.cursor(dictionary=True)
                cursor.execute(sql_query, (id, ))
                history_items = cursor.fetchone() # Fetch all matching rows
                cursor.close()
    except mysql.connector.Error as err:
        print(f"Database Error in get_image_history: {err}")
        st.error("Error fetching image history.")
    
    return history_items

