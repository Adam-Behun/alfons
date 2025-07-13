import logging
from typing import Dict, Any, List, Optional
import requests  # For API calls
# Assuming deepgram-sdk or similar; for MVP, use REST API
# Note: Deepgram and AssemblyAI have Python SDKs, but to avoid extra deps, use requests

from config.settings import settings

logger = logging.getLogger(__name__)

class STTEngine:
    """
    Unified interface for Speech-to-Text (STT) services.
    Supports Deepgram and AssemblyAI; selects based on available API keys.
    Provides transcription with confidence scores and timestamps.
    Handles diarization if supported; else, separate module.
    """
    
    def __init__(self, preferred_engine: str = "deepgram"):
        """
        Initialize STT engine based on available keys.
        
        :param preferred_engine: 'deepgram' or 'assemblyai'.
        """
        self.engine = None
        if settings.DEEPGRAM_API_KEY and preferred_engine == "deepgram":
            self.engine = "deepgram"
        elif settings.ASSEMBLYAI_API_KEY:
            self.engine = "assemblyai"
        else:
            raise ValueError("No valid STT API key found")
        
        logger.info(f"STTEngine initialized with {self.engine}")
    
    def transcribe(self, audio_path: str, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using selected engine.
        
        :param audio_path: Path to the audio file (WAV preferred).
        :param features: Optional features like diarization, language.
        :return: Transcription result with text, confidence, timestamps.
        """
        if self.engine == "deepgram":
            return self._transcribe_deepgram(audio_path, features)
        elif self.engine == "assemblyai":
            return self._transcribe_assemblyai(audio_path, features)
        else:
            raise NotImplementedError(f"Engine {self.engine} not supported")
    
    def _transcribe_deepgram(self, audio_path: str, features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transcribe using Deepgram API.
        
        :param audio_path: Audio file path.
        :param features: Features dict.
        :return: Transcription dict.
        """
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova",
            "diarize": True if features and features.get("diarize") else False,
            "punctuate": True,
            "utterances": True
        }
        
        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(url, headers=headers, params=params, data=audio_file)
            response.raise_for_status()
            result = response.json()
            
            # Format to standard: {'transcript': str, 'segments': list of {'text': str, 'start': float, 'end': float, 'confidence': float, 'speaker': int}}
            transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            segments = []
            for utt in result.get("results", {}).get("utterances", []):
                segments.append({
                    "text": utt["transcript"],
                    "start": utt["start"],
                    "end": utt["end"],
                    "confidence": utt["confidence"],
                    "speaker": utt.get("speaker")
                })
            
            logger.info("Deepgram transcription completed")
            return {"transcript": transcript, "segments": segments}
        except requests.RequestException as e:
            logger.error(f"Deepgram API error: {e}")
            raise
    
    def _transcribe_assemblyai(self, audio_path: str, features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transcribe using AssemblyAI API.
        
        :param audio_path: Audio file path.
        :param features: Features dict.
        :return: Transcription dict.
        """
        # First, upload audio
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {"authorization": settings.ASSEMBLYAI_API_KEY}
        
        try:
            with open(audio_path, "rb") as audio_file:
                upload_resp = requests.post(upload_url, headers=headers, data=audio_file)
            upload_resp.raise_for_status()
            audio_url = upload_resp.json()["upload_url"]
            
            # Transcribe
            transcribe_url = "https://api.assemblyai.com/v2/transcript"
            payload = {
                "audio_url": audio_url,
                "speaker_labels": True if features and features.get("diarize") else False,
                "punctuate": True,
                "format_text": True
            }
            transcribe_resp = requests.post(transcribe_url, json=payload, headers=headers)
            transcribe_resp.raise_for_status()
            transcript_id = transcribe_resp.json()["id"]
            
            # Poll for result
            status_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            while True:
                status_resp = requests.get(status_url, headers=headers)
                status_resp.raise_for_status()
                status = status_resp.json()
                if status["status"] == "completed":
                    break
                elif status["status"] == "error":
                    raise ValueError(f"AssemblyAI error: {status['error']}")
                # Wait logic (simplified for MVP)
                import time
                time.sleep(5)
            
            # Format result
            transcript = status["text"]
            segments = []
            for utt in status.get("utterances", []):
                segments.append({
                    "text": utt["text"],
                    "start": utt["start"] / 1000.0,
                    "end": utt["end"] / 1000.0,
                    "confidence": utt["confidence"],
                    "speaker": utt.get("speaker")
                })
            
            logger.info("AssemblyAI transcription completed")
            return {"transcript": transcript, "segments": segments}
        except requests.RequestException as e:
            logger.error(f"AssemblyAI API error: {e}")
            raise

# Example usage (for testing)
if __name__ == "__main__":
    engine = STTEngine()
    try:
        # Test transcribe (replace with actual file)
        result = engine.transcribe("path/to/sample.wav", {"diarize": True})
        print(f"Transcript: {result['transcript']}")
        print(f"Segments: {len(result['segments'])}")
    except Exception as e:
        print(f"Error: {e}")