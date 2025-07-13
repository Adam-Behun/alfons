import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import os
import json

from config.settings import settings
from ..input_sources.historical_uploader import HistoricalUploader

logger = logging.getLogger(__name__)

app = FastAPI(title="Alfons Upload API")

uploader = HistoricalUploader()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)  # JSON string
):
    """
    Endpoint for uploading MP3/WAV files with optional metadata.
    
    :param file: Uploaded audio file.
    :param metadata: Optional JSON metadata (date, participants, outcome).
    :return: Dict with file_id and message.
    """
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    # Save temp file
    temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    meta_dict = json.loads(metadata) if metadata else None
    
    try:
        file_id = uploader.upload_file(temp_path, meta_dict)
        return {"file_id": file_id, "message": "Upload successful"}
    except Exception as e:
        logger.error(f"Upload API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """
    Get upload/processing status by ID.
    
    :param file_id: File ID.
    :return: Status dict.
    """
    status = uploader.get_upload_status(file_id)
    if not status:
        raise HTTPException(status_code=404, detail="File not found")
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)