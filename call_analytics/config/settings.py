import os
from pydantic import BaseSettings, Field
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
        description="API key for pyannote-audio (if required for Hugging Face access)"
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

# Instantiate settings
settings = Settings()