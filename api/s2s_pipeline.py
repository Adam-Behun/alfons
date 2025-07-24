import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterable
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import audioop
from datetime import datetime

from openai import AsyncOpenAI
import redis.asyncio as redis

from shared.config import config
from shared.logging import get_logger
from .conversation import get_conversation_manager

logger = get_logger(__name__)

class CallState(Enum):
    """Simplified call states."""
    GREETING = "greeting"
    VERIFICATION = "verification"
    AUTH_REQUEST = "auth_request"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    ESCALATION = "escalation"

@dataclass
class ConversationMemory:
    """Conversation memory."""
    call_id: str
    patient_id: Optional[str] = None
    procedure_code: Optional[str] = None
    insurance: Optional[str] = None
    current_state: CallState = CallState.GREETING
    history: List[str] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

class S2SPipeline:
    """
    Simplified S2S pipeline for realtime prior auth calls using OpenAI Realtime API.
    Manages sessions in-memory; basic state/RAG placeholders.
    """

    def __init__(self, model: str = "gpt-4o-realtime-preview-2024-10-01", voice: str = "alloy", enable_rag: bool = True):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model
        self.voice = voice
        self.enable_rag = enable_rag

        self.sessions: Dict[str, Dict[str, Any]] = {}  # call_id: {'memory': Memory, 'realtime_session': connection, 'manager': manager}

        logger.info("S2SPipeline initialized")

    async def start_call_session(self, call_id: str) -> Dict[str, Any]:
        """
        Start session using OpenAI Realtime API via SDK.
        
        :param call_id: ID.
        :return: Session info.
        """
        memory = ConversationMemory(call_id)
        session_config = {
            "modalities": ["text", "audio"],
            "instructions": "You are Alfons, prior auth assistant. Be empathetic, professional.",
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "threshold": 0.5},
            "temperature": 0.7,
            "max_response_output_tokens": 150,
            "input_audio_transcription": {"model": "whisper-1"}
        }

        # Create the context manager and manually enter to get the connection
        realtime_manager = self.client.beta.realtime.connect(model=self.model)
        connection = await realtime_manager.__aenter__()
        
        # Update session configuration
        await connection.session.update(session=session_config)

        self.sessions[call_id] = {
            "memory": memory, 
            "realtime_session": connection, 
            "manager": realtime_manager, 
            "start_time": time.time()
        }

        return {"status": "initialized", "call_id": call_id}

    async def speak(self, call_id: str, text: str):
        """
        Prompt OpenAI to speak a specific text verbatim.

        :param call_id: ID.
        :param text: Text to speak.
        """
        if call_id not in self.sessions:
            raise ValueError("No session")

        connection = self.sessions[call_id]["realtime_session"]
        await connection.response.create(
            response={
                "modalities": ["text", "audio"],
                "instructions": f"Say this verbatim:\n{text}"
            }
        )
        logger.debug(f"Sent speak command for call_id: {call_id}, text: {text}")

    async def process_audio_chunk(self, call_id: str, audio_data: bytes) -> AsyncIterable[bytes]:
        """
        Process chunk: append, generate response.
        
        :param call_id: ID.
        :param audio_data: Mulaw bytes.
        :yield: Response audio (mulaw).
        """
        if call_id not in self.sessions:
            raise ValueError("No session")

        session = self.sessions[call_id]
        connection = session["realtime_session"]
        memory = session["memory"]

        # Convert mulaw to pcm16 at 8kHz, then resample to 24kHz
        pcm_audio = audioop.ulaw2lin(audio_data, 2)
        pcm_audio, _ = audioop.ratecv(pcm_audio, 2, 1, 8000, 24000, None)

        # Append audio to input buffer
        await connection.input_audio_buffer.append(audio=pcm_audio)

        # RAG if enabled
        if self.enable_rag:
            transcript = await self._get_partial_transcript(connection)
            if transcript:
                redis_client = await redis.from_url(config.REDIS_URL)
                await redis_client.publish(f"transcript_{call_id}", json.dumps({"type": "chunk", "role": "user", "chunk": transcript}))
                await redis_client.close()

                rag_context = await self._retrieve_rag(transcript, memory.current_state)
                if rag_context:
                    await connection.session.update(session={"instructions": f"Use context: {rag_context}"})

        # Create response
        await connection.response.create()

        # Handle response events asynchronously
        partial_response_transcript = ""
        async for event in connection:
            if event.type == 'response.audio.delta':
                pcm_chunk = event.audio
                pcm_chunk, _ = audioop.ratecv(pcm_chunk, 2, 1, 24000, 8000, None)
                mulaw_chunk = audioop.lin2ulaw(pcm_chunk, 2)
                yield mulaw_chunk

            elif event.type == 'response.audio_transcript.delta':
                partial_response_transcript += event.transcript
                redis_client = await redis.from_url(config.REDIS_URL)
                await redis_client.publish(f"transcript_{call_id}", json.dumps({"type": "chunk", "role": "assistant", "chunk": event.transcript}))
                await redis_client.close()

            elif event.type == 'response.audio_transcript.done':
                if partial_response_transcript:
                    redis_client = await redis.from_url(config.REDIS_URL)
                    bot_thoughts = await self._get_thoughts(transcript, partial_response_transcript)
                    conv_manager = get_conversation_manager()
                    extracted = await conv_manager._extract_data(partial_response_transcript, None)
                    await redis_client.publish(f"transcript_{call_id}", json.dumps({"type": "complete", "role": "assistant", "thoughts": bot_thoughts, "content": partial_response_transcript, "extracted": extracted, "timestamp": datetime.utcnow().isoformat()}))
                    await redis_client.close()

                    memory.history.append(partial_response_transcript)
                    memory.current_state = self._update_state(memory, partial_response_transcript)

            elif event.type == 'response.done':
                break

            elif event.type == 'error':
                logger.error(f"Realtime error: {event.error.message}")
                raise RuntimeError(event.error.message)

    async def end_call_session(self, call_id: str) -> Dict[str, Any]:
        """
        End session.

        :param call_id: ID.
        :return: Summary.
        """
        if call_id not in self.sessions:
            return {"status": "no_session"}

        session = self.sessions[call_id]
        manager = session["manager"]
        
        # Manually exit the context to close the connection
        await manager.__aexit__(None, None, None)

        # Publish stop event to Redis for WebSocket clients
        redis_client = await redis.from_url(config.REDIS_URL)
        await redis_client.publish(f"transcript_{call_id}", json.dumps({"type": "stop"}))
        await redis_client.close()

        total_time = time.time() - session["start_time"]
        summary = {
            "status": "completed",
            "duration": total_time,
            "state": session["memory"].current_state.value,
            "extracted": {
                "patient_id": session["memory"].patient_id,
                "procedure_code": session["memory"].procedure_code,
                "insurance": session["memory"].insurance
            }
        }

        del self.sessions[call_id]
        return summary

    def _mulaw_to_pcm16(self, mulaw: bytes) -> bytes:
        """Convert mulaw to pcm16 (placeholder)."""
        return mulaw

    def _pcm16_to_mulaw(self, pcm: bytes) -> bytes:
        """Convert pcm16 to mulaw (placeholder)."""
        return pcm

    async def _get_partial_transcript(self, session) -> Optional[str]:
        """Get partial transcript (placeholder)."""
        return "partial text"  # Implement

    async def _retrieve_rag(self, query: str, state: CallState) -> str:
        """Simple RAG retrieve (placeholder)."""
        return "relevant context"  # Integrate Mongo/vector search

    def _update_state(self, memory: ConversationMemory, transcript: str) -> CallState:
        """Update state based on transcript (simple logic)."""
        if "deny" in transcript.lower():
            return CallState.OBJECTION_HANDLING
        return memory.current_state

    async def _get_thoughts(self, message: str, response: str) -> str:
        """Generate thoughts based on message and response."""
        prompt = f"Think step by step about the prior auth task:\nUser message: {message}\nBot response: {response}\nOutput the detailed reasoning thoughts."
        resp = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )
        return resp.choices[0].message.content

def create_s2s_pipeline(enable_rag: bool = True) -> S2SPipeline:
    return S2SPipeline(enable_rag=enable_rag)