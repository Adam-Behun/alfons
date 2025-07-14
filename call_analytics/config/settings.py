import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Configuration settings for the Alfons AI Agent Learning System.
    Uses Pydantic for validation and loading from environment variables.
    Includes API keys, database connections, and other env-specific configs.
    """
    
    # MongoDB Configuration
    MONGO_URI: str = Field(
        default="mongodb://localhost:27017",
        env="MONGO_URI",
        description="MongoDB connection URI"
    )
    MONGO_DB_NAME: str = Field(
        default="alfons_db",
        env="MONGO_DB_NAME",
        description="Name of the MongoDB database"
    )
    
    # API Keys for external services
    DEEPGRAM_API_KEY: Optional[str] = Field(
        default=None,
        env="DEEPGRAM_API_KEY",
        description="API key for Deepgram STT service"
    )
    ASSEMBLYAI_API_KEY: Optional[str] = Field(
        default=None,
        env="ASSEMBLYAI_API_KEY",
        description="API key for AssemblyAI STT service"
    )

    PYANNOTE_API_KEY: Optional[str] = Field(
    default=None,
    env="PYANNOTE_API_KEY",
    description="Hugging Face token for pyannote.audio authentication"
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY", 
        description="API key for OpenAI service"
    )
    
    # Alfons/Twilio Configuration (from main app)
    TWILIO_ACCOUNT_SID: Optional[str] = Field(
        default=None,
        env="TWILIO_ACCOUNT_SID",
        description="Twilio account SID"
    )
    TWILIO_AUTH_TOKEN: Optional[str] = Field(
        default=None,
        env="TWILIO_AUTH_TOKEN",
        description="Twilio auth token"
    )
    TWILIO_PHONE_NUMBER: Optional[str] = Field(
        default=None,
        env="TWILIO_PHONE_NUMBER",
        description="Twilio phone number"
    )
    ELEVENLABS_API_KEY: Optional[str] = Field(
        default=None,
        env="ELEVENLABS_API_KEY",
        description="ElevenLabs API key"
    )
    SUPABASE_URL: Optional[str] = Field(
        default=None,
        env="SUPABASE_URL",
        description="Supabase URL"
    )
    SUPABASE_KEY: Optional[str] = Field(
        default=None,
        env="SUPABASE_KEY",
        description="Supabase key"
    )
    HUMAN_ESCALATION_NUMBER: Optional[str] = Field(
        default=None,
        env="HUMAN_ESCALATION_NUMBER",
        description="Human escalation phone number"
    )
    BASE_URL: Optional[str] = Field(
        default=None,
        env="BASE_URL",
        description="Base URL for webhooks"
    )
    
    # Other configurations
    LOG_LEVEL: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    UPLOAD_DIR: str = Field(
        default="./uploads",
        env="UPLOAD_DIR",
        description="Directory for storing uploaded audio files"
    )
    HIPAA_ANONYMIZE: bool = Field(
        default=True,
        env="HIPAA_ANONYMIZE",
        description="Enable data anonymization for HIPAA-like privacy"
    )
    
    # Model paths or configs (e.g., for embeddings)
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="Sentence-transformers model for embeddings"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env

# Instantiate settings
settings = Settings()