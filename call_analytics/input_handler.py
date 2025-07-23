import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import aiofiles
import aiofiles.os
import requests
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from shared.config import config
from shared.logging import get_logger
from .mongo_connector import AsyncMongoConnector
from .task_queue import process_call

logger = get_logger(__name__)

class InputHandler:
    """
    Handles inputs: Twilio webhooks (live calls) and historical uploads (.mp3 only).
    Queues to Celery for processing; simple metadata/status in MongoDB.
    """

    def __init__(self, upload_dir: str = config.UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.connector = AsyncMongoConnector()
        self.twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        logger.info("InputHandler initialized")

    async def process_twilio_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Twilio webhook: store, download recording if completed, queue processing.

        :param webhook_data: Twilio payload.
        :return: Result dict.
        """
        call_sid = webhook_data.get('CallSid')
        call_status = webhook_data.get('CallStatus')
        logger.info(f"Processing webhook for {call_sid}, status: {call_status}")

        try:
            # Store webhook
            webhook_doc = {
                'call_sid': call_sid,
                'status': call_status,
                'data': webhook_data,
                'received_at': datetime.utcnow().isoformat()
            }
            webhook_id = await self.connector.insert_document("twilio_webhooks", webhook_doc)

            if call_status == 'completed':
                # Fetch details, download recording
                call_details = self._fetch_call_details(call_sid)
                recording_path = self._download_recording(call_sid)

                # Prepare and queue data
                call_data = self._prepare_call_data(webhook_data, call_details, recording_path)
                task = process_call.delay(call_data)
                return {'status': 'queued', 'call_sid': call_sid, 'task_id': task.id}

            return {'status': 'handled', 'call_sid': call_sid}

        except Exception as e:
            logger.error(f"Webhook error for {call_sid}: {e}")
            return {'status': 'error', 'error': str(e)}

    def _fetch_call_details(self, call_sid: str) -> Dict[str, Any]:
        """
        Fetch Twilio call details.

        :param call_sid: SID.
        :return: Details dict.
        """
        try:
            call = self.twilio_client.calls(call_sid).fetch()
            return {
                'from': call.from_,
                'to': call.to,
                'duration': call.duration,
                'start_time': call.start_time.isoformat() if call.start_time else None,
                'end_time': call.end_time.isoformat() if call.end_time else None
            }
        except TwilioRestException as e:
            logger.error(f"Twilio API error: {e}")
            return {}

    def _download_recording(self, call_sid: str) -> Optional[str]:
        """
        Download recording if available.

        :param call_sid: SID.
        :return: Local path or None.
        """
        try:
            recordings = self.twilio_client.recordings.list(call_sid=call_sid)
            if not recordings:
                return None

            recording = recordings[0]
            recording_url = f"https://api.twilio.com{recording.uri.replace('.json', '.mp3')}"
            response = requests.get(recording_url, auth=(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")))

            if response.status_code == 200:
                path = os.path.join(self.upload_dir, f"{call_sid}.mp3")
                with open(path, 'wb') as f:
                    f.write(response.content)
                return path
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    def _prepare_call_data(self, webhook_data: Dict[str, Any], details: Dict[str, Any], recording_path: Optional[str]) -> Dict[str, Any]:
        """
        Prepare data for queue.

        :param webhook_data: Webhook.
        :param details: Call details.
        :param recording_path: Path.
        :return: Data dict.
        """
        return {
            'call_id': webhook_data.get('CallSid'),
            'audio_path': recording_path,
            'details': details,
            'metadata': webhook_data
        }

    async def upload_historical_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload .mp3, store metadata, queue processing.

        :param file_path: Local path.
        :param metadata: Optional metadata.
        :return: File ID.
        """
        if not file_path.lower().endswith('.mp3'):
            raise ValueError("Only .mp3 supported")

        dest_path = None
        try:
            unique_filename = f"{uuid.uuid4()}.mp3"
            dest_path = os.path.join(self.upload_dir, unique_filename)
            
            async with aiofiles.open(file_path, 'rb') as src:
                content = await src.read()
            async with aiofiles.open(dest_path, 'wb') as dst:
                await dst.write(content)

            file_meta = {
                'upload_date': datetime.utcnow().isoformat(),
                'file_path': dest_path,
                'original_name': metadata.get('original_name', Path(file_path).name),
                'processed': False,
                'outcome': metadata.get('outcome', 'pending'),
                'metadata': metadata or {},
                'status': 'uploaded'
            }
            if metadata and 'patient_name' in metadata:
                file_meta['metadata']['patient_name'] = 'REDACTED'  # HIPAA-compliant anonymization

            file_id = await self.connector.insert_document("historical_uploads", file_meta)

            call_data = {'call_id': file_id, 'audio_path': dest_path, 'metadata': file_meta}
            task = process_call.delay(call_data)
            await self.connector.update_document("historical_uploads", {"_id": file_id}, {"$set": {"task_id": task.id, "status": "queued"}})

            logger.info(f"Uploaded {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"Upload error: {e}")
            if dest_path and await aiofiles.os.path.exists(dest_path):
                await aiofiles.os.unlink(dest_path)
            raise

    async def get_status(self, file_id: str) -> Dict[str, Any]:
        """
        Get upload/processing status.

        :param file_id: ID.
        :return: Status dict.
        """
        doc = await self.connector.find_documents("historical_uploads", {"_id": file_id}, limit=1)
        return doc[0] if doc else {}

    async def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Cleanup old files.

        :param days_old: Age threshold.
        :return: Cleaned count.
        """
        cutoff = datetime.utcnow().timestamp() - (days_old * 86400)
        count = 0
        for file in os.listdir(self.upload_dir):
            path = os.path.join(self.upload_dir, file)
            if os.path.isfile(path) and os.path.getctime(path) < cutoff:
                await aiofiles.os.unlink(path)
                count += 1
        return count