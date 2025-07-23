import asyncio
import logging
import time
from typing import Dict, Any, Union, AsyncIterable, Optional
from pathlib import Path
import os
import tempfile

from .s2s_pipeline import create_s2s_pipeline
from call_analytics.transcription_pipeline import TranscriptionPipeline

from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)

class SpeechProcessingRouter:
    """
    Simplified speech router: delegates realtime to S2S, batch to transcription pipeline.
    Auto-detects mode based on input; uses providers internally.
    """

    def __init__(self):
        self.s2s_pipeline = create_s2s_pipeline()
        self.transcription_pipeline = TranscriptionPipeline()
        self.stats = {"realtime_requests": 0, "batch_requests": 0, "total_time": 0.0, "avg_latency": 0.0}
        logger.info("SpeechProcessingRouter initialized")

    async def process_speech(
        self,
        audio_input: Union[str, bytes, AsyncIterable[bytes]],
        mode: str = "auto",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process speech: auto-detect mode (realtime for streams, batch for files/bytes).

        :param audio_input: Path, bytes, or stream.
        :param mode: 'realtime', 'batch', 'auto'.
        :param context: Optional context.
        :return: Results dict.
        """
        start_time = time.time()
        context = context or {}

        if mode == "auto":
            mode = "realtime" if isinstance(audio_input, AsyncIterable) else "batch"

        logger.info(f"Processing with mode: {mode}")

        if mode == "realtime":
            result = await self._process_realtime(audio_input, context)
            self.stats["realtime_requests"] += 1
        elif mode == "batch":
            result = await self._process_batch(audio_input, context)
            self.stats["batch_requests"] += 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        processing_time = time.time() - start_time
        self.stats["total_time"] += processing_time
        total_requests = self.stats["realtime_requests"] + self.stats["batch_requests"]
        self.stats["avg_latency"] = self.stats["total_time"] / total_requests if total_requests else 0.0

        result["processing"] = {"mode": mode, "time": processing_time}
        return result

    async def _process_realtime(
        self,
        audio_input: Union[bytes, AsyncIterable[bytes]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Realtime via S2S pipeline.

        :param audio_input: Bytes or stream.
        :param context: Context.
        :return: Results.
        """
        call_id = context.get("call_id", f"speech_{int(time.time())}")
        await self.s2s_pipeline.start_call_session(call_id)

        if isinstance(audio_input, AsyncIterable):
            responses = []
            async for chunk in audio_input:
                async for resp_chunk in self.s2s_pipeline.process_audio_chunk(call_id, chunk):
                    responses.append(resp_chunk)
            result = {"type": "realtime_stream", "call_id": call_id, "response_chunks": len(responses)}
        else:
            responses = []
            async for resp_chunk in self.s2s_pipeline.process_audio_chunk(call_id, audio_input):
                responses.append(resp_chunk)
            result = {"type": "realtime_single", "call_id": call_id, "response_audio": responses[0] if responses else None}

        summary = await self.s2s_pipeline.end_call_session(call_id)
        result["summary"] = summary
        return result

    async def _process_batch(
        self,
        audio_input: Union[str, bytes],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Batch via transcription pipeline.

        :param audio_input: Path or bytes.
        :param context: Context.
        :return: Results.
        """
        if isinstance(audio_input, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                temp.write(audio_input)
                audio_path = temp.name
            cleanup = True
        else:
            audio_path = audio_input
            cleanup = False

        transcript = await self.transcription_pipeline.transcribe_audio(audio_path)

        result = {"type": "batch", "transcript": transcript, "status": "completed"}

        if cleanup:
            os.unlink(audio_path)

        return result

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check.

        :return: Health dict.
        """
        health = {"status": "healthy", "components": {}}
        # Simple checks
        health["components"]["s2s"] = "healthy"
        health["components"]["transcription"] = "healthy"
        return health

def get_speech_router() -> SpeechProcessingRouter:
    return SpeechProcessingRouter()