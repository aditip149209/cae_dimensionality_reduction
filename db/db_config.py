import mysql.connector
from contextlib import contextmanager
import streamlit as st
import os

@contextmanager
def get_db_connection():
    """
    Provides a robust database connection as a context manager.
    It handles acquisition, yielding, and guaranteed cleanup.
    """
    # 1. Check for secrets BEFORE trying to connect
    try:
        if "DB_HOST" not in st.secrets:
            st.error("Database secrets not found. Please check your .streamlit/secrets.toml file.")
            # We must raise an error to stop execution
            raise ValueError("Missing database secrets")
        
        db_config = {
            "host": st.secrets["DB_HOST"],
            "user": st.secrets["DB_USER"],
            "password": st.secrets["DB_PASS"],
            "database": st.secrets["DB_NAME"]
        }
    except KeyError as e:
        st.error(f"Missing a required secret: {e}. Please check your .streamlit/secrets.toml")
        raise
        
    conn = None
    try:
        # 2. ACQUIRE the connection
        conn = mysql.connector.connect(**db_config)
        
        # 3. YIELD the connection (the one and only yield)
        yield conn
        
    except mysql.connector.Error as err:
        # 4. HANDLE errors (e.g., connection failed or query inside 'with' failed)
        st.error(f"Database Error: {err}")
        print(f"Database Error: {err}")
        # Re-raise the exception to stop the app from proceeding
        raise
        
    finally:
        # 5. CLEANUP (this *always* runs)
        if conn and conn.is_connected():
            conn.close()

