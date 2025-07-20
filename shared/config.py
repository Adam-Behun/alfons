# Centralized configuration management for Alfons AI Voice Agent
# Consolidates settings from call_analytics/config/settings.py and env variables

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"

class LLMProvider(str, Enum):
    OPENAI = "openai"

class STTProvider(str, Enum):
    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"

class TTSProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"

class Settings(BaseSettings):
    """
    Unified configuration settings for the Alfons AI Agent system.
    Merges existing settings with new provider abstractions.
    """
    
    # === Core Service Configuration ===
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    BASE_URL: Optional[str] = Field(default=None, env="BASE_URL")
    HIPAA_ANONYMIZE: bool = Field(default=True, env="HIPAA_ANONYMIZE")
    
    # === Provider Selection ===
    LLM_PROVIDER: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    STT_PROVIDER: STTProvider = Field(default=STTProvider.OPENAI, env="STT_PROVIDER") 
    TTS_PROVIDER: TTSProvider = Field(default=TTSProvider.ELEVENLABS, env="TTS_PROVIDER")
    
    # === Database Configuration ===
    # MongoDB (primary)
    MONGO_URI: str = Field(
        default="mongodb://localhost:27017",
        env="MONGO_URI"
    )
    MONGO_DB_NAME: str = Field(default="alfons_db", env="MONGO_DB_NAME")
    
    # === Redis Configuration ===
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # === Telephony Configuration ===
    TWILIO_ACCOUNT_SID: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    HUMAN_ESCALATION_NUMBER: Optional[str] = Field(default=None, env="HUMAN_ESCALATION_NUMBER")
    
    # === AI Service API Keys ===
    # LLM Providers
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # STT Providers
    DEEPGRAM_API_KEY: Optional[str] = Field(default=None, env="DEEPGRAM_API_KEY")
    ASSEMBLYAI_API_KEY: Optional[str] = Field(default=None, env="ASSEMBLYAI_API_KEY")
    
    # TTS Providers
    ELEVENLABS_API_KEY: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # Other AI Services
    PYANNOTE_API_KEY: Optional[str] = Field(default=None, env="PYANNOTE_API_KEY")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    
    # === Model Configuration ===
    # LLM Models
    OPENAI_MODEL: str = Field(default="gpt-4o-mini-2024-07-18", env="OPENAI_MODEL")
    
    # TTS Configuration
    ELEVENLABS_VOICE_ID: str = Field(default="21m00Tcm4TlvDQ8ikWAM", env="ELEVENLABS_VOICE_ID")
    ELEVENLABS_VOICE_SETTINGS: Dict[str, Any] = Field(
        default={
            "stability": 0.7,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    )
    
    # Embedding Model
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # === Performance Configuration ===
    # S2S Pipeline
    S2S_TARGET_LATENCY_MS: int = Field(default=500, env="S2S_TARGET_LATENCY_MS")
    STT_CHUNK_SIZE: int = Field(default=1024, env="STT_CHUNK_SIZE")
    TTS_CHUNK_SIZE: int = Field(default=1024, env="TTS_CHUNK_SIZE")
    
    # Request Timeouts
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    
    # === File Management ===
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    STATIC_DIR: str = Field(default="./static", env="STATIC_DIR")
    AUDIO_CLEANUP_HOURS: int = Field(default=1, env="AUDIO_CLEANUP_HOURS")
    
    # === Queue Configuration ===
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"
        validate_assignment = True
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider"""
        key_mapping = {
            "openai": self.OPENAI_API_KEY,
            "elevenlabs": self.ELEVENLABS_API_KEY,
            "deepgram": self.DEEPGRAM_API_KEY,
            "assemblyai": self.ASSEMBLYAI_API_KEY,
            "pyannote": self.PYANNOTE_API_KEY
        }
        return key_mapping.get(provider.lower())
    
    def validate_provider_config(self) -> Dict[str, bool]:
        """Validate that selected providers have required API keys"""
        results = {}
        
        # Check LLM provider
        llm_key = self.get_api_key(self.LLM_PROVIDER.value)
        results[f"llm_{self.LLM_PROVIDER.value}"] = llm_key is not None
        
        # Check STT provider  
        stt_key = self.get_api_key(self.STT_PROVIDER.value)
        results[f"stt_{self.STT_PROVIDER.value}"] = stt_key is not None
        
        # Check TTS provider
        tts_key = self.get_api_key(self.TTS_PROVIDER.value)
        results[f"tts_{self.TTS_PROVIDER.value}"] = tts_key is not None
        
        return results
    
    def validate_required_config(self) -> Dict[str, bool]:
        """Validate that all required configuration is present."""
        validation_results = {}
        
        # Required API keys based on selected providers
        validation_results["openai_api_key"] = bool(self.OPENAI_API_KEY)
        validation_results["twilio_config"] = all([
            self.TWILIO_ACCOUNT_SID,
            self.TWILIO_AUTH_TOKEN,
            self.TWILIO_PHONE_NUMBER
        ])
        
        # Database configuration
        validation_results["mongodb_config"] = bool(self.MONGO_URI)
        validation_results["redis_config"] = bool(self.REDIS_URL)
        
        # Provider-specific validation
        provider_validation = self.validate_provider_config()
        validation_results.update(provider_validation)
        
        return validation_results

# Global settings instance
config = Settings()

# Validate configuration on import
try:
    validation_results = config.validate_required_config()
    missing_config = [key for key, valid in validation_results.items() if not valid]
    
    if missing_config:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Missing configuration for: {', '.join(missing_config)}")
        logger.warning("Some features may not work properly. Check your .env file.")
        
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Configuration validation failed: {e}")

# Export commonly used values for convenience
MONGO_DB_NAME = config.MONGO_DB_NAME
MONGO_URI = config.MONGO_URI
OPENAI_API_KEY = config.OPENAI_API_KEY
REDIS_URL = config.REDIS_URL