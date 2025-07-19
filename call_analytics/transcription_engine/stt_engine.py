"""
Unified Speech-to-Text engine supporting multiple providers with real-time and batch processing.
Implements provider abstraction for OpenAI Whisper, Deepgram, and AssemblyAI.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import aiohttp
import json
import time
from pathlib import Path

from shared.config import config
from shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionSegment:
    """Standardized transcription segment structure."""
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    transcript: str
    segments: List[TranscriptionSegment]
    duration: float
    language: Optional[str] = None
    processing_time: float = 0.0


class ISTTProvider(ABC):
    """Abstract interface for STT providers."""
    
    @abstractmethod
    async def transcribe_batch(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file (batch processing)."""
        pass
    
    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """Transcribe audio stream (real-time processing)."""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports real-time streaming."""
        pass


class OpenAIWhisperProvider(ISTTProvider):
    """OpenAI Whisper STT provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/audio"
        
    async def transcribe_batch(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using OpenAI Whisper."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('file', open(audio_path, 'rb'), filename=Path(audio_path).name)
        data.add_field('model', kwargs.get('model', 'whisper-1'))
        data.add_field('response_format', 'verbose_json')
        data.add_field('timestamp_granularities[]', 'segment')
        
        if kwargs.get('language'):
            data.add_field('language', kwargs['language'])
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/transcriptions",
                    headers=headers,
                    data=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            # Convert to standard format
            segments = []
            for segment in result.get('segments', []):
                segments.append(TranscriptionSegment(
                    text=segment['text'].strip(),
                    start=segment['start'],
                    end=segment['end'],
                    confidence=1.0,  # Whisper doesn't provide confidence scores
                    speaker=None
                ))
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                transcript=result.get('text', ''),
                segments=segments,
                duration=result.get('duration', 0.0),
                language=result.get('language'),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """OpenAI Whisper doesn't support real-time streaming."""
        raise NotImplementedError("OpenAI Whisper doesn't support real-time streaming")
    
    def supports_streaming(self) -> bool:
        return False


class DeepgramProvider(ISTTProvider):
    """Deepgram STT provider with real-time and batch support."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1"
        
    async def transcribe_batch(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using Deepgram."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"
        }
        
        params = {
            "model": kwargs.get('model', 'nova-2'),
            "diarize": kwargs.get('diarize', False),
            "punctuate": True,
            "utterances": True,
            "smart_format": True
        }
        
        if kwargs.get('language'):
            params['language'] = kwargs['language']
            
        try:
            async with aiohttp.ClientSession() as session:
                with open(audio_path, 'rb') as audio_file:
                    async with session.post(
                        f"{self.base_url}/listen",
                        headers=headers,
                        params=params,
                        data=audio_file.read()
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
            
            # Convert to standard format
            channel = result.get("results", {}).get("channels", [{}])[0]
            transcript = channel.get("alternatives", [{}])[0].get("transcript", "")
            
            segments = []
            for utt in result.get("results", {}).get("utterances", []):
                segments.append(TranscriptionSegment(
                    text=utt["transcript"].strip(),
                    start=utt["start"],
                    end=utt["end"],
                    confidence=utt["confidence"],
                    speaker=f"SPEAKER_{utt.get('speaker', 0)}" if utt.get('speaker') is not None else None
                ))
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                transcript=transcript,
                segments=segments,
                duration=channel.get("alternatives", [{}])[0].get("duration", 0.0),
                language=result.get("results", {}).get("language"),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """Real-time transcription using Deepgram streaming."""
        # Implementation would use WebSocket connection to Deepgram streaming API
        # For now, placeholder that shows the interface
        yield TranscriptionSegment(
            text="[Streaming not implemented]",
            start=0.0,
            end=1.0,
            confidence=1.0
        )
    
    def supports_streaming(self) -> bool:
        return True


class AssemblyAIProvider(ISTTProvider):
    """AssemblyAI STT provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        
    async def transcribe_batch(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using AssemblyAI."""
        start_time = time.time()
        
        headers = {"authorization": self.api_key}
        
        try:
            # Upload audio
            async with aiohttp.ClientSession() as session:
                with open(audio_path, 'rb') as audio_file:
                    async with session.post(
                        f"{self.base_url}/upload",
                        headers=headers,
                        data=audio_file.read()
                    ) as response:
                        response.raise_for_status()
                        upload_result = await response.json()
                        audio_url = upload_result["upload_url"]
                
                # Submit transcription
                payload = {
                    "audio_url": audio_url,
                    "speaker_labels": kwargs.get('diarize', False),
                    "punctuate": True,
                    "format_text": True
                }
                
                if kwargs.get('language'):
                    payload['language_code'] = kwargs['language']
                
                async with session.post(
                    f"{self.base_url}/transcript",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    submit_result = await response.json()
                    transcript_id = submit_result["id"]
                
                # Poll for completion
                while True:
                    async with session.get(
                        f"{self.base_url}/transcript/{transcript_id}",
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        status = await response.json()
                        
                        if status["status"] == "completed":
                            break
                        elif status["status"] == "error":
                            raise ValueError(f"AssemblyAI error: {status.get('error', 'Unknown error')}")
                        
                        await asyncio.sleep(2)  # Poll every 2 seconds
            
            # Convert to standard format
            segments = []
            for utt in status.get("utterances", []):
                segments.append(TranscriptionSegment(
                    text=utt["text"].strip(),
                    start=utt["start"] / 1000.0,  # Convert ms to seconds
                    end=utt["end"] / 1000.0,
                    confidence=utt["confidence"],
                    speaker=f"SPEAKER_{utt.get('speaker', 'A')}" if utt.get('speaker') else None
                ))
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                transcript=status.get("text", ""),
                segments=segments,
                duration=status.get("audio_duration", 0.0),
                language=status.get("language_code"),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """AssemblyAI real-time transcription."""
        # Implementation would use WebSocket connection to AssemblyAI streaming API
        yield TranscriptionSegment(
            text="[Streaming not implemented]",
            start=0.0,
            end=1.0,
            confidence=1.0
        )
    
    def supports_streaming(self) -> bool:
        return True


class STTEngine:
    """Unified STT engine with provider abstraction and healthcare optimizations."""
    
    def __init__(self, preferred_provider: str = "deepgram"):
        self.provider = self._initialize_provider(preferred_provider)
        logger.info(f"STTEngine initialized with {type(self.provider).__name__}")
    
    def _initialize_provider(self, preferred: str) -> ISTTProvider:
        """Initialize STT provider based on available API keys."""
        providers_config = {
            "openai": (config.OPENAI_API_KEY, OpenAIWhisperProvider),
            "deepgram": (config.DEEPGRAM_API_KEY, DeepgramProvider),
            "assemblyai": (config.ASSEMBLYAI_API_KEY, AssemblyAIProvider)
        }
        
        # Try preferred provider first
        if preferred in providers_config:
            api_key, provider_class = providers_config[preferred]
            if api_key:
                return provider_class(api_key)
        
        # Fallback to any available provider
        for name, (api_key, provider_class) in providers_config.items():
            if api_key:
                logger.warning(f"Preferred provider '{preferred}' not available, using {name}")
                return provider_class(api_key)
        
        raise ValueError("No STT provider API keys available")
    
    async def transcribe_batch(
        self,
        audio_path: str,
        diarize: bool = True,
        language: Optional[str] = None,
        model: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file with healthcare-specific optimizations.
        
        Args:
            audio_path: Path to audio file
            diarize: Enable speaker diarization
            language: Target language (auto-detect if None)
            model: Specific model to use
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        logger.info(f"Starting batch transcription for {audio_path}")
        
        kwargs = {
            "diarize": diarize,
            "language": language,
            "model": model
        }
        
        try:
            result = await self.provider.transcribe_batch(audio_path, **kwargs)
            logger.info(f"Transcription completed in {result.processing_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            raise
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        **kwargs
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """
        Real-time transcription for streaming audio.
        
        Args:
            audio_stream: Async generator yielding audio chunks
            **kwargs: Provider-specific options
            
        Yields:
            TranscriptionSegment objects as they become available
        """
        if not self.provider.supports_streaming():
            raise NotImplementedError(f"{type(self.provider).__name__} doesn't support streaming")
        
        logger.info("Starting real-time transcription")
        
        async for segment in self.provider.transcribe_stream(audio_stream, **kwargs):
            yield segment
    
    def supports_streaming(self) -> bool:
        """Check if current provider supports real-time streaming."""
        return self.provider.supports_streaming()