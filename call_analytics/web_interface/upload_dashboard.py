import streamlit as st
import logging
import os
from typing import Dict, Any

from config.settings import settings
from ..input_sources.historical_uploader import HistoricalUploader
from ..database.mongo_connector import MongoConnector

logger = logging.getLogger(__name__)

def run_upload_dashboard():
    """
    Streamlit dashboard for file uploads and status monitoring.
    Allows drag-and-drop of MP3/WAV files, metadata input, and displays upload/processing status.
    """
    st.title("Alfons Historical Call Upload Dashboard")
    
    uploader = HistoricalUploader()
    connector = MongoConnector()
    
    try:
        # File uploader
        uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
        
        if uploaded_file:
            # Save temp file
            temp_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Metadata inputs
            st.subheader("Metadata")
            date = st.date_input("Call Date")
            participants = st.text_input("Participants (e.g., rep, insurance)")
            outcome = st.selectbox("Outcome", ["success", "failure", "pending"])
            
            metadata: Dict[str, Any] = {
                "date": date.isoformat() if date else None,
                "participants": participants,
                "outcome": outcome
            }
            
            if st.button("Submit Upload"):
                try:
                    file_id = uploader.upload_file(temp_path, metadata)
                    st.success(f"File uploaded successfully! ID: {file_id}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    logger.error(f"Upload error: {e}")
    
        # Status display
        st.subheader("Upload Status")
        uploads = connector.find_documents("uploads", limit=10)  # Recent 10
        for upload in uploads:
            st.write(f"ID: {upload['_id']}, File: {upload['original_name']}, Processed: {upload['processed']}, Outcome: {upload.get('outcome', 'N/A')}")
    
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")
    finally:
        connector.close_connection()

if __name__ == "__main__":
    run_upload_dashboard()