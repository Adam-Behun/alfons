import asyncio
import json
import logging
import base64
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import time
import audioop

from fastapi import WebSocketDisconnect
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from .s2s_pipeline import S2SPipeline, create_s2s_pipeline
from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)

class EnhancedTelephonyService:
    """
    Simplified telephony service for real-time S2S streaming with Twilio Media Streams.
    Focuses on outbound calls, WebSocket handling, and integration with S2S pipeline.
    """

    def __init__(self):
        # Twilio client
        self.twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

        # S2S pipeline
        self.s2s_pipeline = create_s2s_pipeline()

        # Config
        self.base_url = config.BASE_URL
        self.twilio_phone = config.TWILIO_PHONE_NUMBER

        # Active calls (simplified tracking)
        self.active_calls: Dict[str, Dict[str, Any]] = {}

        # Stream params
        self.sample_rate = 8000
        self.chunk_size = 160  # 20ms at 8kHz

        logger.info("EnhancedTelephonyService initialized (streaming only)")

    def make_call(self, phone_number: str, call_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Initiate outbound call with streaming.

        :param phone_number: E.164 format.
        :param call_context: Optional context.
        :return: Call SID.
        """
        logger.info(f"Making streaming call to {phone_number}")

        try:
            webhook_url = f"{self.base_url}/voice/streaming"
            call = self.twilio_client.calls.create(
                to=phone_number,
                from_=self.twilio_phone,
                url=webhook_url,
                method="POST"
            )

            self.active_calls[call.sid] = {
                "phone_number": phone_number,
                "context": call_context or {},
                "start_time": datetime.utcnow(),
                "status": "initiated"
            }

            logger.info(f"Call initiated: {call.sid}")
            return call.sid

        except Exception as e:
            logger.error(f"Call failed: {e}")
            raise

    def generate_streaming_twiml(self, call_sid: str) -> str:
        """
        Generate TwiML for streaming calls.

        :param call_sid: Call SID.
        :return: TwiML string.
        """
        response = VoiceResponse()
        response.say("Hi, this is Alfons. Calling regarding a prior authorization for our patient", voice="alice")

        connect = Connect()
        stream = Stream(url=f"wss://{self.base_url.replace('https://', '')}/ws/media-stream")
        connect.append(stream)
        response.append(connect)

        return str(response)

    async def handle_stream_connection(self, websocket, path: str):
        """
        Handle Twilio Media Stream WebSocket.

        :param websocket: Connection.
        :param path: Path.
        """
        call_id = None
        stream_id = str(uuid.uuid4())  # Simple ID

        try:
            while True:
                message = await websocket.receive_text()
                event = json.loads(message)
                event_type = event.get("event")

                if event_type == "start":
                    call_id = event.get("start", {}).get("callSid")
                    if call_id:
                        await self.s2s_pipeline.start_call_session(call_id)
                        logger.info(f"Started session for {call_id}")

                elif event_type == "media" and call_id:
                    await self._process_media(event, websocket, call_id)

                elif event_type == "stop" and call_id:
                    await self.s2s_pipeline.end_call_session(call_id)
                    logger.info(f"Ended session for {call_id}")
                    break  # Exit loop on stream stop

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            if call_id and call_id in self.active_calls:
                del self.active_calls[call_id]

    async def _process_media(self, event: Dict[str, Any], websocket, call_id: str):
        """
        Process incoming audio media.

        :param event: Media event.
        :param websocket: Connection.
        :param call_id: Call ID.
        """
        payload = event.get("media", {}).get("payload")
        if payload:
            audio_data = base64.b64decode(payload)
            pcm_audio = audioop.ulaw2lin(audio_data, 2)
            pcm_audio, _ = audioop.ratecv(pcm_audio, 2, 1, 8000, 24000, None)
            async for response_audio in self.s2s_pipeline.process_audio_chunk(call_id, pcm_audio):
                pcm_chunk, _ = audioop.ratecv(response_audio, 2, 1, 24000, 8000, None)
                mulaw_chunk = audioop.lin2ulaw(pcm_chunk, 2)
                await self._send_response(websocket, event["streamSid"], mulaw_chunk)

    async def _send_response(self, websocket, stream_sid: str, audio_data: bytes):
        """
        Send audio back to Twilio.

        :param websocket: Connection.
        :param stream_sid: Stream SID.
        :param audio_data: Bytes.
        """
        payload = base64.b64encode(audio_data).decode("utf-8")
        media_event = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        await websocket.send_text(json.dumps(media_event))

    async def get_service_health(self) -> Dict[str, Any]:
        """
        Health check.

        :return: Health dict.
        """
        health = {"status": "healthy", "active_calls": len(self.active_calls)}
        try:
            self.twilio_client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
            health["twilio"] = "healthy"
        except Exception as e:
            health["status"] = "degraded"
            health["twilio"] = str(e)
        return health

# Global instance (inject via deps in main.py)
telephony_service: Optional[EnhancedTelephonyService] = None

def get_telephony_service() -> EnhancedTelephonyService:
    global telephony_service
    if telephony_service is None:
        telephony_service = EnhancedTelephonyService()
    return telephony_service