# Abstract interface for Text-to-Speech providers  
# Enables swapping between ElevenLabs, OpenAI TTS, etc.

from abc import ABC, abstractmethod
from typing import Optional, Union, AsyncGenerator
from pydantic import BaseModel
import asyncio

class TTSResponse(BaseModel):
    """Standardized TTS response format"""
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    format: str = "mp3"
    provider: Optional[str] = None
    latency_ms: Optional[float] = None
    character_count: Optional[int] = None

class StreamingTTSResponse(BaseModel):
    """Standardized streaming TTS response"""
    audio_chunk: bytes
    is_final: bool = False
    chunk_index: int = 0

class VoiceSettings(BaseModel):
    """Voice configuration settings"""
    voice_id: str
    stability: float = 0.7
    similarity_boost: float = 0.8
    style: float = 0.0
    use_speaker_boost: bool = True

class ITTSProvider(ABC):
    """
    Abstract interface for Text-to-Speech providers.
    Supports both batch and streaming synthesis.
    """
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        voice_settings: Optional[VoiceSettings] = None,
        output_format: str = "mp3",
        **kwargs
    ) -> TTSResponse:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice_settings: Voice configuration
            output_format: Audio format (mp3, wav, etc.)
            **kwargs: Provider-specific parameters
            
        Returns:
            TTSResponse with audio data/URL and metadata
        """
        pass
    
    @abstractmethod
    async def stream_synthesize(
        self,
        text: str,
        voice_settings: Optional[VoiceSettings] = None,
        output_format: str = "mp3", 
        **kwargs
    ) -> AsyncGenerator[StreamingTTSResponse, None]:
        """
        Stream synthesized speech as it's generated
        
        Args:
            text: Text to convert
            voice_settings: Voice configuration
            output_format: Audio format
            **kwargs: Provider-specific parameters
            
        Yields:
            StreamingTTSResponse chunks
        """
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> list[dict]:
        """
        Get list of available voices from provider
        
        Returns:
            List of voice dictionaries with id, name, language, etc.
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
    def supports_custom_voices(self) -> bool:
        """Whether this provider supports voice cloning/customization"""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of supported audio formats"""
        pass
    
    def optimize_for_healthcare(self, text: str) -> str:
        """
        Optimize text for healthcare TTS with pronunciation guides
        
        Args:
            text: Original text
            
        Returns:
            Text optimized for medical terminology pronunciation
        """
        # Common medical abbreviations and their pronunciations
        medical_replacements = {
            "CPT": "C P T",
            "ICD": "I C D", 
            "mg": "milligrams",
            "ml": "milliliters",
            "mcg": "micrograms",
            "qd": "once daily",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily"
        }
        
        optimized_text = text
        for abbrev, pronunciation in medical_replacements.items():
            optimized_text = optimized_text.replace(abbrev, pronunciation)
        
        return optimized_text
    
    def add_phonetic_spelling(self, text: str, alphanumeric_strings: list[str]) -> str:
        """
        Add phonetic spelling for patient IDs, auth numbers, etc.
        
        Args:
            text: Original text
            alphanumeric_strings: List of strings to make phonetic
            
        Returns:
            Text with phonetic spelling added
        """
        phonetic_alphabet = {
            'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta',
            'E': 'Echo', 'F': 'Foxtrot', 'G': 'Golf', 'H': 'Hotel',
            'I': 'India', 'J': 'Juliet', 'K': 'Kilo', 'L': 'Lima',
            'M': 'Mike', 'N': 'November', 'O': 'Oscar', 'P': 'Papa',
            'Q': 'Quebec', 'R': 'Romeo', 'S': 'Sierra', 'T': 'Tango',
            'U': 'Uniform', 'V': 'Victor', 'W': 'Whiskey', 'X': 'X-ray',
            'Y': 'Yankee', 'Z': 'Zulu'
        }
        
        result_text = text
        for string in alphanumeric_strings:
            if string in text:
                phonetic_parts = []
                for char in string.upper():
                    if char.isalpha():
                        phonetic_parts.append(f"{char} as in {phonetic_alphabet.get(char, char)}")
                    elif char.isdigit():
                        phonetic_parts.append(char)
                    else:
                        phonetic_parts.append(char)
                
                phonetic_version = ", ".join(phonetic_parts)
                result_text = result_text.replace(string, f"{string}, that's {phonetic_version}")
        
        return result_text