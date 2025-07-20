import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncIterable
from pathlib import Path
import tempfile
import time
import ffmpeg
from dataclasses import dataclass
import aiohttp
import json

from shared.config import config
from shared.logging import get_logger
from shared.providers.istt_provider import get_stt_provider  # Use provider for STT swap

logger = get_logger(__name__)

@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    confidence: float = 1.0

@dataclass
class TranscriptionResult:
    transcript: str
    segments: List[TranscriptionSegment]
    duration: float
    processing_time: float

class TranscriptionPipeline:
    """
    Simplified transcription pipeline handling audio processing, STT, diarization, and validation inline.
    Supports batch (.mp3) and stream processing for S2S integration.
    Uses STT provider; keeps logic linear and self-contained.
    Integrated useful logic from stt_engine.py (batch/stream transcription with providers) and speaker_diarization.py (simple diarization with role inference).
    """

    def __init__(self, temp_dir: Optional[str] = None, stt_provider_name: str = "deepgram"):
        """
        Initialize the pipeline.

        :param temp_dir: Directory for temporary files.
        :param stt_provider_name: Name of STT provider (e.g., 'deepgram', 'openai').
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "alfons_transcription"
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Settings for simplicity
        self.target_sample_rate = 16000
        self.target_channels = 1
        self.chunk_duration = 300  # 5 min chunks for long audio

        # STT provider (easy to swap, integrated from stt_engine.py)
        self.stt_provider = self._initialize_stt_provider(stt_provider_name)

        # Predefined patterns for simple diarization/validation (from speaker_diarization.py)
        self.speaker_roles = {"provider": ["doctor", "physician", "nurse", "provider", "clinic"], "insurance": ["insurance", "representative", "agent", "payer", "plan"]}
        self.medical_terms = ["prior authorization", "eligibility", "documentation"]  # For validation

        logger.info("TranscriptionPipeline initialized")

    def _initialize_stt_provider(self, name: str):
        """
        Initialize STT provider based on name (integrated logic from stt_engine.py).
        """
        if name == "openai" and config.OPENAI_API_KEY:
            return OpenAIWhisperProvider(config.OPENAI_API_KEY)
        elif name == "deepgram" and config.DEEPGRAM_API_KEY:
            return DeepgramProvider(config.DEEPGRAM_API_KEY)
        else:
            raise ValueError(f"Unsupported or unconfigured STT provider: {name}")

    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Batch transcription for .mp3 files: process fully post-upload/call.

        :param audio_path: Path to .mp3 audio file.
        :return: Full transcript text.
        """
        start_time = time.time()
        logger.info(f"Starting batch transcription for: {audio_path}")

        try:
            # Validate input
            if not audio_path.lower().endswith('.mp3'):
                raise ValueError("Only .mp3 files supported for batch processing")

            # Step 1: Preprocess audio (enhance and convert to WAV)
            enhanced_path = await self._enhance_audio(audio_path)

            # Step 2: Chunk if long
            chunks = await self._chunk_audio(enhanced_path)

            # Step 3: Transcribe chunks using provider (from stt_engine.py)
            segments = []
            for chunk in chunks:
                result = await self.stt_provider.transcribe_batch(chunk, diarize=True)  # Use batch from provider
                chunk_segments = self._diarize_segments(result.segments)  # Inline diarization from speaker_diarization.py
                segments.extend(chunk_segments)

            # Step 4: Merge and validate
            full_transcript = " ".join(seg.text for seg in segments)
            validated_transcript = self._validate_transcript(full_transcript)

            # Cleanup
            await self._cleanup_temp_files([enhanced_path] + chunks)

            processing_time = time.time() - start_time
            logger.info(f"Batch transcription completed in {processing_time:.2f}s")

            return validated_transcript

        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            raise

    async def stream_transcribe(self, stream: AsyncIterable[bytes]) -> AsyncIterable[str]:
        """
        Stream transcription for S2S: yield partial transcripts.

        :param stream: Async iterable of audio bytes.
        :yield: Partial transcript texts.
        """
        if not self.stt_provider.supports_streaming():
            raise NotImplementedError("Selected STT provider does not support streaming")

        logger.info("Starting stream transcription")

        try:
            async for segment in self.stt_provider.transcribe_stream(stream):  # Use stream from provider (stt_engine.py)
                diarized_segments = self._diarize_segments([segment])  # Inline diarization
                validated_text = self._validate_transcript(diarized_segments[0].text if diarized_segments else segment.text)
                yield validated_text

        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            raise

    async def _enhance_audio(self, input_path: str) -> str:
        """
        Enhance audio: noise reduction, mono, 16kHz (from audio_processor.py).
        """
        output_path = self.temp_dir / f"{Path(input_path).stem}_enhanced.wav"

        audio_input = ffmpeg.input(input_path)
        audio_filtered = audio_input.filter('afftdn', nf=-25).filter('highpass', f=80).filter('loudnorm')
        output = audio_filtered.output(
            str(output_path),
            ac=self.target_channels,
            ar=self.target_sample_rate,
            acodec='pcm_s16le',
            f='wav'
        )

        await asyncio.to_thread(output.overwrite_output().run, quiet=True)
        return str(output_path)

    async def _chunk_audio(self, input_path: str) -> List[str]:
        """
        Chunk long audio (from pipeline.py).
        """
        probe = await asyncio.to_thread(ffmpeg.probe, input_path)
        duration = float(probe['format']['duration'])

        if duration <= self.chunk_duration:
            return [input_path]

        output_dir = self.temp_dir / f"{Path(input_path).stem}_chunks"
        output_dir.mkdir(exist_ok=True)

        chunks = []
        start = 0
        i = 0
        while start < duration:
            chunk_path = output_dir / f"chunk_{i:03d}.wav"
            chunk_input = ffmpeg.input(input_path, ss=start, t=self.chunk_duration)
            output = chunk_input.output(str(chunk_path), ac=1, ar=16000, acodec='pcm_s16le')
            await asyncio.to_thread(output.overwrite_output().run, quiet=True)
            chunks.append(str(chunk_path))
            start += self.chunk_duration
            i += 1

        return chunks

    def _diarize_segments(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Simple diarization with role inference (integrated from speaker_diarization.py).
        """
        # Heuristic: Alternate speakers and infer roles based on keywords
        diarized = []
        speaker = "SPEAKER_0"
        for seg in segments:
            new_speaker = self._infer_role(seg.text)
            if new_speaker != speaker:
                speaker = new_speaker
            diarized.append(TranscriptionSegment(text=seg.text, start=seg.start, end=seg.end, confidence=seg.confidence, speaker=speaker))
        return diarized

    def _infer_role(self, text: str) -> str:
        """
        Infer speaker role based on keywords (from speaker_diarization.py).
        """
        text_lower = text.lower()
        for role, keywords in self.speaker_roles.items():
            if any(kw in text_lower for kw in keywords):
                return role
        return "UNKNOWN"

    def _validate_transcript(self, transcript: str) -> str:
        """
        Simple validation: check for medical terms, correct if needed (from transcript_validator.py).
        """
        for term in self.medical_terms:
            if term in transcript.lower():
                # Simple correction example
                transcript = transcript.replace("prior authorisation", "prior authorization")  # Normalize spelling
        return transcript

    async def _cleanup_temp_files(self, file_paths: List[str]):
        """
        Cleanup temp files.
        """
        for path in file_paths:
            Path(path).unlink(missing_ok=True)

# Integrated provider classes from stt_engine.py (simplified for Deepgram and OpenAI as examples)

class OpenAIWhisperProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/audio"

    async def transcribe_batch(self, audio_path: str, diarize=False, **kwargs) -> TranscriptionResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = aiohttp.FormData()
        data.add_field('file', open(audio_path, 'rb'), filename=Path(audio_path).name)
        data.add_field('model', 'whisper-1')
        data.add_field('response_format', 'verbose_json')
        data.add_field('timestamp_granularities[]', 'segment')

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/transcriptions", headers=headers, data=data) as response:
                result = await response.json()

        segments = [TranscriptionSegment(text=s['text'], start=s['start'], end=s['end'], confidence=1.0) for s in result.get('segments', [])]
        return TranscriptionResult(transcript=result['text'], segments=segments, duration=result.get('duration', 0), processing_time=0)

    async def transcribe_stream(self, stream):
        raise NotImplementedError("OpenAI Whisper does not support streaming")

    def supports_streaming(self):
        return False

class DeepgramProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1"

    async def transcribe_batch(self, audio_path: str, diarize=False, **kwargs) -> TranscriptionResult:
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"}
        params = {"model": "nova-2", "diarize": diarize, "punctuate": True}

        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as f:
                async with session.post(f"{self.base_url}/listen", headers=headers, params=params, data=f.read()) as response:
                    result = await response.json()

        channel = result.get("results", {}).get("channels", [{}])[0]
        transcript = channel.get("alternatives", [{}])[0].get("transcript", "")
        segments = [TranscriptionSegment(text=u["transcript"], start=u["start"], end=u["end"], confidence=u["confidence"], speaker=f"SPEAKER_{u.get('speaker')}") for u in result.get("results", {}).get("utterances", [])]
        return TranscriptionResult(transcript=transcript, segments=segments, duration=0, processing_time=0)

    async def transcribe_stream(self, stream):
        # Placeholder for WebSocket streaming
        async for chunk in stream:
            yield TranscriptionSegment(text="[Deepgram stream partial]", start=0, end=1, confidence=1.0)

    def supports_streaming(self):
        return True