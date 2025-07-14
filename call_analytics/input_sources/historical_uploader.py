import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from multiprocessing import Queue  # For queuing; could use celery for production

from call_analytics.config.settings import settings
from ..database.mongo_connector import MongoConnector

logger = logging.getLogger(__name__)

class HistoricalUploader:
    """
    Handles historical call file uploads, metadata capture, and queuing for processing.
    Supports MP3/WAV files, stores files locally, metadata in MongoDB, and triggers pipelines.
    For MVP, processing is synchronous; queue for async in future.
    Ensures HIPAA-like anonymization by not storing sensitive data directly.
    """
    
    def __init__(self, upload_dir: str = settings.UPLOAD_DIR):
        """
        Initialize uploader with directory and MongoDB connector.
        
        :param upload_dir: Directory to store uploaded files.
        """
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.connector = MongoConnector()
        self.processing_queue = Queue()  # Simple queue for triggering pipelines
        
        logger.info(f"HistoricalUploader initialized with upload_dir: {self.upload_dir}")
    
    def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a file, store it locally, capture metadata, and queue for processing.
        
        :param file_path: Path to the MP3/WAV file.
        :param metadata: Optional metadata (date, participants, outcome).
        :return: Uploaded file ID (MongoDB insert ID).
        """
        if not file_path.lower().endswith(('.mp3', '.wav')):
            raise ValueError("Only MP3 or WAV files are supported")
        
        # Copy file to upload_dir
        base_name = os.path.basename(file_path)
        dest_path = os.path.join(self.upload_dir, base_name)
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        
        # Default metadata
        default_metadata = {
            "upload_date": datetime.utcnow().isoformat(),
            "file_path": dest_path,
            "original_name": base_name,
            "processed": False
        }
        if metadata:
            default_metadata.update(metadata)
        
        # Anonymize if enabled
        if settings.HIPAA_ANONYMIZE:
            self._anonymize_metadata(default_metadata)
        
        # Insert metadata to MongoDB
        file_id = self.connector.insert_document("uploads", default_metadata)
        
        # Queue for processing (for MVP, process synchronously)
        self._process_file(dest_path, file_id)
        
        logger.info(f"Uploaded file {base_name} with ID: {file_id}")
        return file_id
    
    def _anonymize_metadata(self, metadata: Dict[str, Any]):
        """
        Anonymize sensitive fields in metadata.
        
        :param metadata: Metadata to anonymize (modified in-place).
        """
        # Example: Redact participants if they contain names
        if 'participants' in metadata:
            metadata['participants'] = 'REDACTED'
        # Add more as needed
    
    def _process_file(self, file_path: str, file_id: str):
        """
        Trigger processing pipeline (transcription, analysis, etc.).
        For MVP: Placeholder; integrate actual pipeline calls here.
        
        :param file_path: Path to the uploaded file.
        :param file_id: ID of the upload document.
        """
        # TODO: Integrate with transcription/audio_processor.py, etc.
        logger.info(f"Processing file {file_path} (ID: {file_id})")
        # Example: Update status
        self.connector.update_document("uploads", {"_id": file_id}, {"$set": {"processed": True}})
    
    def get_upload_status(self, file_id: str) -> Dict[str, Any]:
        """
        Get status of an uploaded file.
        
        :param file_id: ID of the upload.
        :return: Metadata document.
        """
        docs = self.connector.find_documents("uploads", {"_id": file_id}, limit=1)
        return docs[0] if docs else None

# Example usage (for testing)
if __name__ == "__main__":
    uploader = HistoricalUploader()
    try:
        # Test upload (replace with actual file path)
        test_metadata = {"participants": "rep, insurance", "outcome": "success"}
        file_id = uploader.upload_file("path/to/sample.mp3", test_metadata)
        print(f"Uploaded file ID: {file_id}")
        
        status = uploader.get_upload_status(file_id)
        print(f"Status: {status}")
    finally:
        uploader.connector.close_connection()