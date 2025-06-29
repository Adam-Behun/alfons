"""
speech.py

Module handles speech-to-text and text-to-speech operations using the ElevenLabs API
for the Alfons backend.

https://elevenlabs.io/docs/api-reference/introduction
Environment variables required:
- ELEVENLABS_API_KEY: API key for ElevenLabs
- BASE_URL: The public URL where generated audio files can be accessed

Functions:
- transcribe_audio: Converts audio at a given URL to text using ElevenLabs
- synthesize_speech: Converts text to speech, saves the audio, and returns a public URL
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()


async def transcribe_audio(audio_url: str) -> str:
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
    data = {"audio_url": audio_url}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["transcript"]


async def synthesize_speech(text: str) -> str:
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDQ8ikWAm"
    headers = {
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8}
    }
    response = requests.post(url, headers=headers, json=data)
    with open("static/output.mp3", "wb") as f:
        f.write(response.content)
    return f"{os.getenv('BASE_URL')}/static/output.mp3"
