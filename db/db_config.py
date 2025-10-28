import os
import mysql.connector
import streamlit as st
from contextlib import contextmanager

@contextmanager
def get_db_connection():    
    db_config = {}

    # 1. Try to get config from Environment Variables (for Docker)
    # We use the 'DB_HOST', 'DB_USER' etc. from your provided code.
    db_config['host'] = os.environ.get('DB_HOST') 
    db_config['user'] = os.environ.get('DB_USER')
    db_config['password'] = os.environ.get('DB_PASS')
    db_config['database'] = os.environ.get('DB_NAME')

    conn = None
    try:
        # 3. ACQUIRE the connection
        conn = mysql.connector.connect(**db_config)
        
        # 4. YIELD the connection to the 'with' block
        yield conn
        
    except mysql.connector.Error as err:
        # 5. HANDLE any errors that occurred
        st.error(f"Database Error: {err}")
        print(f"Database Error: {err}")
        # Re-raise the exception to stop the app
        raise
        
    finally:
        # 6. CLEANUP (this *always* runs, even if errors happen)
        if conn and conn.is_connected():
            conn.close()
