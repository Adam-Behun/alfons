"""
speech.py

Speech-to-text (Human to text for OpenAI) and text-to-speech (OpenAI to speech for human) 
using OpenAI Whisper and ElevenLabs APIs for the Alfons backend.

Environment variables required:
- OPENAI_API_KEY: API key for OpenAI Whisper
- ELEVENLABS_API_KEY: API key for ElevenLabs TTS
- TWILIO_ACCOUNT_SID: Twilio account SID for authenticated audio downloads
- TWILIO_AUTH_TOKEN: Twilio auth token for authenticated audio downloads
- BASE_URL: The public URL where generated audio files can be accessed

Functions:
- transcribe_audio: Converts audio at a given URL to text using OpenAI Whisper
- synthesize_speech: Converts text to speech, saves the audio, and returns a public URL
"""

import requests
import os
from dotenv import load_dotenv
import logging
import time
from typing import Optional
import aiohttp
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDQ8ikWAM"  # Rachel voice
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

async def transcribe_audio(audio_url: str) -> str:
    """
    Transcribe audio from URL using OpenAI Whisper API with robust error handling
    """
    logger.info(f"Starting transcription for: {audio_url}")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not openai_api_key:
        logger.error("OpenAI API key not found")
        return "[API key not configured]"
    
    if not twilio_sid or not twilio_token:
        logger.error("Twilio credentials not found")
        return "[Twilio credentials not configured]"
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Transcription attempt {attempt + 1}/{MAX_RETRIES}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
                # Download the audio file from Twilio with authentication
                auth = aiohttp.BasicAuth(twilio_sid, twilio_token)
                
                async with session.get(audio_url, auth=auth) as audio_response:
                    if audio_response.status != 200:
                        logger.error(f"Failed to download audio: {audio_response.status}")
                        if attempt == MAX_RETRIES - 1:
                            return "[Audio download failed]"
                        await asyncio.sleep(1)
                        continue
                    
                    audio_data = await audio_response.read()
                    logger.info(f"Downloaded audio file, size: {len(audio_data)} bytes")
                
                # Send to OpenAI Whisper API
                url = "https://api.openai.com/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {openai_api_key}"}
                
                form_data = aiohttp.FormData()
                form_data.add_field('file', audio_data, filename='recording.wav', content_type='audio/wav')
                form_data.add_field('model', 'whisper-1')
                form_data.add_field('response_format', 'text')
                
                async with session.post(url, headers=headers, data=form_data) as response:
                    logger.info(f"Transcription API response status: {response.status}")
                    
                    if response.status == 200:
                        result = await response.text()
                        transcript = result.strip()
                        logger.info(f"Transcription successful: {transcript}")
                        
                        if transcript:
                            return transcript
                        else:
                            logger.warning("Empty transcript returned")
                            return "[No speech detected]"
                    
                    elif response.status == 429:
                        logger.warning("Rate limit exceeded, waiting before retry...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Transcription API error {response.status}: {error_text}")
                        
                        if attempt == MAX_RETRIES - 1:
                            return "[Transcription service error]"
                        
                        await asyncio.sleep(1)
                        
        except asyncio.TimeoutError:
            logger.error(f"Transcription timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                return "[Transcription timeout]"
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Transcription error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return "[Error during transcription]"
            await asyncio.sleep(1)
    
    return "[Transcription failed after retries]"

async def synthesize_speech(text: str) -> Optional[str]:
    """
    Convert text to speech using ElevenLabs API with robust error handling
    Returns URL to generated audio file or None if failed
    """
    logger.info(f"Starting speech synthesis for: {text[:100]}...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.error("ElevenLabs API key not found")
        return None
    
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not configured")
        return None
    
    # Ensure static directory exists
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    
    # Create unique filename to avoid conflicts
    timestamp = int(time.time())
    filename = f"speech_{timestamp}.mp3"
    filepath = os.path.join(static_dir, filename)
    
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{DEFAULT_VOICE_ID}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text[:1000],  # Limit text length
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Speech synthesis attempt {attempt + 1}/{MAX_RETRIES}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    logger.info(f"Speech synthesis API response status: {response.status}")
                    
                    if response.status == 200:
                        # Save the audio file
                        audio_content = await response.read()
                        
                        with open(filepath, "wb") as f:
                            f.write(audio_content)
                        
                        audio_url = f"{base_url}/static/{filename}"
                        logger.info(f"Speech generated successfully: {audio_url}")
                        
                        # Verify file was created and has content
                        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                            return audio_url
                        else:
                            logger.error("Generated audio file is empty or doesn't exist")
                            return None
                    
                    elif response.status == 429:
                        logger.warning("Rate limit exceeded, waiting before retry...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Speech synthesis API error {response.status}: {error_text}")
                        
                        if attempt == MAX_RETRIES - 1:
                            return None
                        
                        await asyncio.sleep(1)
                        
        except asyncio.TimeoutError:
            logger.error(f"Speech synthesis timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                return None
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Speech synthesis error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return None
            await asyncio.sleep(1)
    
    return None

def cleanup_old_audio_files():
    """Clean up old audio files to prevent disk space issues"""
    try:
        static_dir = "static"
        if not os.path.exists(static_dir):
            return
        
        current_time = time.time()
        for filename in os.listdir(static_dir):
            if filename.startswith("speech_") and filename.endswith(".mp3"):
                filepath = os.path.join(static_dir, filename)
                file_age = current_time - os.path.getctime(filepath)
                
                # Delete files older than 1 hour
                if file_age > 3600:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audio file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to delete old audio file {filename}: {e}")
                        
    except Exception as e:
        logger.error(f"Error during audio cleanup: {e}")

# Run cleanup when module is imported
cleanup_old_audio_files()