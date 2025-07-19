# Abstract interface for Speech-to-Text providers
# Enables swapping between OpenAI Whisper, Deepgram, AssemblyAI, etc.

from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator, Union
from pydantic import BaseModel
import asyncio

class STTResponse(BaseModel):
    """Standardized STT response format"""
    transcript: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    provider: Optional[str] = None
    latency_ms: Optional[float] = None
    word_timestamps: Optional[list] = None

class StreamingSTTResponse(BaseModel):
    """Standardized streaming STT response"""
    transcript: str
    is_final: bool = False
    confidence: Optional[float] = None
    word_timestamps: Optional[list] = None

class ISTTProvider(ABC):
    """
    Abstract interface for Speech-to-Text providers.
    Supports both batch and streaming transcription.
    """
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, str],
        language: str = "en",
        model: Optional[str] = None,
        **kwargs
    ) -> STTResponse:
        """
        Transcribe audio data or URL to text
        
        Args:
            audio_data: Audio bytes or URL to audio file
            language: Language code (e.g., "en", "es")
            model: Specific model to use (provider-dependent)
            **kwargs: Provider-specific parameters
            
        Returns:
            STTResponse with transcript and metadata
        """
        pass
    
    @abstractmethod
    async def stream_transcribe(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en",
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamingSTTResponse, None]:
        """
        Transcribe streaming audio in real-time
        
        Args:
            audio_stream: Async generator of audio chunks
            language: Language code
            model: Specific model to use
            **kwargs: Provider-specific parameters
            
        Yields:
            StreamingSTTResponse with partial and final transcripts
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Test if the provider connection is working"""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports real-time streaming"""
        pass
    
    @property
    @abstractmethod
    def supports_speaker_diarization(self) -> bool:
        """Whether this provider supports speaker separation"""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of supported audio formats (e.g., ['wav', 'mp3', 'flac'])"""
        pass
    
    def prepare_healthcare_audio(self, audio_data: bytes, enhance_medical: bool = True) -> bytes:
        """
        Prepare audio for healthcare transcription with medical terminology enhancement
        
        Args:
            audio_data: Raw audio bytes
            enhance_medical: Whether to apply medical audio enhancement
            
        Returns:
            Processed audio bytes optimized for medical terminology
        """
        # Base implementation - providers can override for specific optimizations
        return audio_data
    
    def validate_audio_quality(self, audio_data: bytes) -> dict:
        """
        Validate audio quality for transcription
        
        Args:
            audio_data: Audio bytes to validate
            
        Returns:
            Dict with quality metrics (volume, clarity, duration, etc.)
        """
        return {
            "size_bytes": len(audio_data),
            "valid": len(audio_data) > 0,
            "estimated_duration": None  # Providers can implement actual analysis
        }