"""
Enhanced telephony module with Twilio Media Streams for real-time S2S integration.
Supports both traditional webhooks and streaming WebSocket connections.
"""

import asyncio
import json
import logging
import base64
from typing import List, Dict, Any, Optional
import websockets
from datetime import datetime

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Stream

from .s2s_pipeline import S2SPipeline, create_s2s_pipeline
from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


class TwilioMediaStreamHandler:
    """Handles Twilio Media Streams WebSocket connections for real-time audio."""
    
    def __init__(self, s2s_pipeline: S2SPipeline):
        """
        Initialize Media Stream handler.
        
        Args:
            s2s_pipeline: S2S pipeline for processing audio
        """
        self.s2s_pipeline = s2s_pipeline
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        
        # Twilio Media Stream configuration
        self.sample_rate = 8000  # 8kHz for phone calls
        self.encoding = "mulaw"  # μ-law encoding from Twilio
        self.chunk_size = 160    # 20ms chunks at 8kHz
        
        logger.info("TwilioMediaStreamHandler initialized")
    
    async def handle_stream_connection(self, websocket, path: str):
        """
        Handle incoming WebSocket connection from Twilio Media Stream.
        
        Args:
            websocket: WebSocket connection
            path: WebSocket path (contains call info)
        """
        call_id = None
        stream_id = None
        
        try:
            logger.info(f"New Media Stream connection on path: {path}")
            
            async for message in websocket:
                try:
                    event = json.loads(message)
                    event_type = event.get("event")
                    
                    if event_type == "connected":
                        logger.info("Media Stream connected")
                        
                    elif event_type == "start":
                        # Extract call and stream info
                        call_id = event.get("start", {}).get("callSid")
                        stream_id = event.get("streamSid")
                        
                        if call_id:
                            # Initialize S2S session
                            session_info = await self.s2s_pipeline.start_call_session(call_id)
                            
                            # Store stream info
                            self.active_streams[stream_id] = {
                                "call_id": call_id,
                                "websocket": websocket,
                                "start_time": datetime.utcnow(),
                                "audio_buffer": b"",
                                "sequence_number": 0
                            }
                            
                            logger.info(f"Started S2S session for call {call_id}, stream {stream_id}")
                        
                    elif event_type == "media":
                        # Process incoming audio
                        if stream_id in self.active_streams:
                            await self._process_media_event(event, stream_id)
                        
                    elif event_type == "stop":
                        # Stream ended
                        if stream_id in self.active_streams:
                            await self._handle_stream_stop(stream_id)
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for stream {stream_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Cleanup on connection close
            if stream_id and stream_id in self.active_streams:
                await self._handle_stream_stop(stream_id)
    
    async def _process_media_event(self, event: Dict[str, Any], stream_id: str):
        """
        Process incoming media (audio) event from Twilio.
        
        Args:
            event: Media event from Twilio
            stream_id: Stream identifier
        """
        try:
            stream_info = self.active_streams[stream_id]
            call_id = stream_info["call_id"]
            
            # Extract audio data
            media = event.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode base64 audio (μ-law format)
            audio_data = base64.b64decode(payload)
            
            # Add to buffer for processing in chunks
            stream_info["audio_buffer"] += audio_data
            
            # Process when we have enough data (e.g., 200ms worth)
            buffer_size = len(stream_info["audio_buffer"])
            min_chunk_size = self.sample_rate * 0.2  # 200ms at 8kHz = 1600 bytes
            
            if buffer_size >= min_chunk_size:
                # Process chunk through S2S pipeline
                chunk_to_process = stream_info["audio_buffer"][:int(min_chunk_size)]
                stream_info["audio_buffer"] = stream_info["audio_buffer"][int(min_chunk_size):]
                
                # Send to S2S pipeline and get response
                async for response_audio in self.s2s_pipeline.process_audio_chunk(
                    call_id, chunk_to_process
                ):
                    # Send response back to Twilio
                    await self._send_audio_response(stream_id, response_audio)
        
        except Exception as e:
            logger.error(f"Error processing media event: {e}")
    
    async def _send_audio_response(self, stream_id: str, audio_data: bytes):
        """
        Send audio response back to Twilio Media Stream.
        
        Args:
            stream_id: Stream identifier
            audio_data: Audio data to send (μ-law format)
        """
        try:
            if stream_id not in self.active_streams:
                return
            
            stream_info = self.active_streams[stream_id]
            websocket = stream_info["websocket"]
            
            # Encode audio as base64
            payload = base64.b64encode(audio_data).decode("utf-8")
            
            # Create media event
            media_event = {
                "event": "media",
                "streamSid": stream_id,
                "media": {
                    "payload": payload
                }
            }
            
            # Send to Twilio
            await websocket.send(json.dumps(media_event))
            
            stream_info["sequence_number"] += 1
        
        except Exception as e:
            logger.error(f"Error sending audio response: {e}")
    
    async def _handle_stream_stop(self, stream_id: str):
        """
        Handle stream stop/cleanup.
        
        Args:
            stream_id: Stream identifier
        """
        try:
            if stream_id not in self.active_streams:
                return
            
            stream_info = self.active_streams[stream_id]
            call_id = stream_info["call_id"]
            
            # End S2S session
            if call_id:
                summary = await self.s2s_pipeline.end_call_session(call_id)
                logger.info(f"S2S session ended for call {call_id}: {summary}")
            
            # Cleanup
            del self.active_streams[stream_id]
            
            logger.info(f"Stream {stream_id} cleanup completed")
        
        except Exception as e:
            logger.error(f"Error handling stream stop: {e}")


class EnhancedTelephonyService:
    """
    Enhanced telephony service supporting both traditional webhooks and real-time streaming.
    Integrates Twilio calling with S2S pipeline for low-latency conversations.
    """
    
    def __init__(self, enable_streaming: bool = True):
        """
        Initialize enhanced telephony service.
        
        Args:
            enable_streaming: Enable Media Streams for real-time processing
        """
        # Initialize Twilio client
        self.twilio_client = Client(
            config.TWILIO_ACCOUNT_SID, 
            config.TWILIO_AUTH_TOKEN
        )
        
        # Initialize S2S pipeline
        self.s2s_pipeline = create_s2s_pipeline(
            enable_rag=True
        )
        
        # Initialize Media Stream handler
        self.stream_handler = TwilioMediaStreamHandler(self.s2s_pipeline) if enable_streaming else None
        
        # Configuration
        self.base_url = config.BASE_URL
        self.twilio_phone_number = config.TWILIO_PHONE_NUMBER
        self.enable_streaming = enable_streaming
        
        # Call tracking
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"EnhancedTelephonyService initialized (streaming: {enable_streaming})")
    
    def make_call(
        self, 
        phone_number: str, 
        use_streaming: bool = None,
        call_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Initiate an outbound call with optional real-time streaming.
        
        Args:
            phone_number: Destination phone number in E.164 format
            use_streaming: Use Media Streams (defaults to service setting)
            call_context: Additional context for the call
            
        Returns:
            Call SID
        """
        logger.info(f"Making call to {phone_number}")
        
        # Determine if streaming should be used
        use_streaming = use_streaming if use_streaming is not None else self.enable_streaming
        
        try:
            # Choose webhook URL based on streaming preference
            if use_streaming and self.stream_handler:
                webhook_url = f"{self.base_url}/voice/streaming"
            else:
                webhook_url = f"{self.base_url}/voice"
            
            # Make the call
            call = self.twilio_client.calls.create(
                to=phone_number,
                from_=self.twilio_phone_number,
                url=webhook_url,
                method="POST"
            )
            
            # Track call info
            self.active_calls[call.sid] = {
                "phone_number": phone_number,
                "use_streaming": use_streaming,
                "context": call_context or {},
                "start_time": datetime.utcnow(),
                "status": "initiated"
            }
            
            logger.info(f"Call initiated: {call.sid} (streaming: {use_streaming})")
            return call.sid
            
        except Exception as e:
            logger.error(f"Failed to make call to {phone_number}: {e}")
            raise
    
    def generate_streaming_twiml(self, call_sid: str) -> str:
        """
        Generate TwiML for streaming-enabled calls.
        
        Args:
            call_sid: Call SID
            
        Returns:
            TwiML response string
        """
        response = VoiceResponse()
        
        # Add initial greeting
        response.say(
            "Hello, this is Alfons, your prior authorization assistant. "
            "Please hold while I connect you to our real-time system.",
            voice="alice"
        )
        
        # Start Media Stream
        start = Start()
        stream = Stream(
            url=f"wss://{config.BASE_URL.replace('https://', '').replace('http://', '')}/ws/media-stream"
        )
        start.append(stream)
        response.append(start)
        
        # Add pause to allow stream setup
        response.pause(length=1)
        
        # Continue with conversation
        response.say(
            "I'm ready to assist you with your prior authorization request. How may I help you today?",
            voice="alice"
        )
        
        return str(response)
    
    def generate_webhook_twiml(self, call_sid: str, audio_url: Optional[str] = None) -> str:
        """
        Generate TwiML for traditional webhook-based calls.
        
        Args:
            call_sid: Call SID
            audio_url: URL of recorded audio (for processing)
            
        Returns:
            TwiML response string
        """
        response = VoiceResponse()
        
        if audio_url:
            # Process recorded audio (existing logic)
            # This would integrate with the existing speech processing
            response.say(
                "Thank you for your message. I'm processing your request.",
                voice="alice"
            )
            # Add logic to process audio and respond
        else:
            # Initial call setup
            response.say(
                "Welcome to Alfons, your prior authorization assistant. "
                "Please tell me your patient ID, procedure code, and insurance information after the beep.",
                voice="alice"
            )
            
            response.record(
                action=f"{self.base_url}/voice",
                max_length=30,
                play_beep=True,
                timeout=5,
                transcribe=False  # We'll handle transcription
            )
        
        return str(response)
    
    async def handle_call_status_update(self, call_sid: str, status: str, **kwargs):
        """
        Handle call status updates from Twilio webhooks.
        
        Args:
            call_sid: Call SID
            status: New call status
            **kwargs: Additional status information
        """
        logger.info(f"Call {call_sid} status update: {status}")
        
        try:
            if call_sid in self.active_calls:
                call_info = self.active_calls[call_sid]
                call_info["status"] = status
                
                # Handle specific status changes
                if status == "completed":
                    await self._handle_call_completion(call_sid, call_info)
                elif status == "failed":
                    await self._handle_call_failure(call_sid, call_info, kwargs)
                elif status == "busy" or status == "no-answer":
                    await self._handle_call_unsuccessful(call_sid, call_info, status)
        
        except Exception as e:
            logger.error(f"Error handling status update for {call_sid}: {e}")
    
    async def _handle_call_completion(self, call_sid: str, call_info: Dict[str, Any]):
        """Handle successful call completion."""
        try:
            # Calculate call duration
            duration = datetime.utcnow() - call_info["start_time"]
            
            # Ensure S2S session is properly ended
            if call_info.get("use_streaming", False):
                summary = await self.s2s_pipeline.end_call_session(call_sid)
                logger.info(f"S2S session summary for {call_sid}: {summary}")
            
            # Log call completion
            logger.info(f"Call {call_sid} completed successfully (duration: {duration})")
            
            # Trigger post-call processing (analytics, learning, etc.)
            await self._trigger_post_call_processing(call_sid, call_info)
            
        except Exception as e:
            logger.error(f"Error handling call completion for {call_sid}: {e}")
        finally:
            # Cleanup call tracking
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]
    
    async def _handle_call_failure(self, call_sid: str, call_info: Dict[str, Any], error_info: Dict[str, Any]):
        """Handle call failure."""
        logger.error(f"Call {call_sid} failed: {error_info}")
        
        try:
            # Cleanup any active sessions
            if call_info.get("use_streaming", False):
                await self.s2s_pipeline.end_call_session(call_sid)
            
            # Log failure details
            logger.error(f"Call failure details: {error_info}")
            
        except Exception as e:
            logger.error(f"Error handling call failure for {call_sid}: {e}")
        finally:
            # Cleanup
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]
    
    async def _handle_call_unsuccessful(self, call_sid: str, call_info: Dict[str, Any], reason: str):
        """Handle unsuccessful calls (busy, no answer, etc.)."""
        logger.info(f"Call {call_sid} unsuccessful: {reason}")
        
        try:
            # Cleanup any active sessions
            if call_info.get("use_streaming", False):
                await self.s2s_pipeline.end_call_session(call_sid)
            
            # Could implement retry logic here based on reason
            # For now, just log
            logger.info(f"Call {call_sid} ended with reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error handling unsuccessful call for {call_sid}: {e}")
        finally:
            # Cleanup
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]
    
    async def _trigger_post_call_processing(self, call_sid: str, call_info: Dict[str, Any]):
        """
        Trigger post-call processing for analytics and learning.
        
        Args:
            call_sid: Call SID
            call_info: Call information
        """
        try:
            # This would trigger the learning pipeline
            # Integration point with call_analytics from previous phases
            
            processing_data = {
                "call_sid": call_sid,
                "phone_number": call_info["phone_number"],
                "duration": (datetime.utcnow() - call_info["start_time"]).total_seconds(),
                "used_streaming": call_info.get("use_streaming", False),
                "context": call_info.get("context", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Queue for processing (would integrate with task queue from Phase 1)
            logger.info(f"Queuing post-call processing for {call_sid}")
            
            # Placeholder for actual queue integration
            # await task_queue.enqueue("process_completed_call", processing_data)
            
        except Exception as e:
            logger.error(f"Error triggering post-call processing for {call_sid}: {e}")
    
    def get_call_status(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a call.
        
        Args:
            call_sid: Call SID
            
        Returns:
            Call status information or None if not found
        """
        if call_sid in self.active_calls:
            call_info = self.active_calls[call_sid].copy()
            call_info["duration"] = (datetime.utcnow() - call_info["start_time"]).total_seconds()
            return call_info
        return None
    
    def get_active_calls(self) -> List[Dict[str, Any]]:
        """Get list of all active calls."""
        active = []
        for call_sid, call_info in self.active_calls.items():
            info = call_info.copy()
            info["call_sid"] = call_sid
            info["duration"] = (datetime.utcnow() - info["start_time"]).total_seconds()
            active.append(info)
        return active
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of telephony service and dependencies."""
        health = {
            "status": "healthy",
            "components": {},
            "active_calls": len(self.active_calls),
            "streaming_enabled": self.enable_streaming
        }
        
        try:
            # Check Twilio connectivity
            try:
                account = self.twilio_client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
                health["components"]["twilio"] = {
                    "status": "healthy",
                    "account_status": account.status
                }
            except Exception as e:
                health["components"]["twilio"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check S2S pipeline health
            try:
                pipeline_status = self.s2s_pipeline.get_pipeline_status() if hasattr(self.s2s_pipeline, 'get_pipeline_status') else {"status": "unknown"}
                health["components"]["s2s_pipeline"] = pipeline_status
            except Exception as e:
                health["components"]["s2s_pipeline"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check Media Stream handler
            if self.stream_handler:
                health["components"]["media_streams"] = {
                    "status": "healthy",
                    "active_streams": len(self.stream_handler.active_streams)
                }
            else:
                health["components"]["media_streams"] = {
                    "status": "disabled"
                }
        
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health


# Global service instance (will be initialized in main.py)
telephony_service: Optional[EnhancedTelephonyService] = None


def get_telephony_service() -> EnhancedTelephonyService:
    """Get the global telephony service instance."""
    global telephony_service
    if telephony_service is None:
        telephony_service = EnhancedTelephonyService()
    return telephony_service


# Legacy function compatibility
def make_call(phone_number: str) -> str:
    """
    Legacy function for backward compatibility.
    
    Args:
        phone_number: Phone number to call
        
    Returns:
        Call SID
    """
    service = get_telephony_service()
    return service.make_call(phone_number, use_streaming=False)  # Default to webhook mode for compatibility