import logging
from typing import Dict, List, Any, Optional
from pyannote.audio import Pipeline
import json
import torch  # For device management if needed

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """
    Handles speaker diarization using pyannote-audio.
    Labels speakers and outputs as JSON format.
    Requires Hugging Face token for pretrained models.
    Integrates with audio files post-processing.
    """
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", use_auth_token: Optional[str] = settings.PYANNOTE_API_KEY):
        """
        Initialize the diarization pipeline.
        
        :param model_name: Pretrained model from Hugging Face.
        :param use_auth_token: Hugging Face API token.
        """
        try:
            self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            logger.info(f"SpeakerDiarizer initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> str:
        """
        Perform speaker diarization on an audio file.
        
        :param audio_path: Path to the audio file (WAV).
        :param num_speakers: Optional number of speakers (if known).
        :return: JSON string with diarization results.
        """
        try:
            diarization = self.pipeline(audio_path, num_speakers=num_speakers)
            
            # Convert to JSON format: list of {'speaker': str, 'start': float, 'end': float}
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "speaker": f"SPEAKER_{speaker}",
                    "start": turn.start,
                    "end": turn.end
                })
            
            json_output = json.dumps(results, indent=4)
            logger.info(f"Diarization completed for {audio_path}")
            return json_output
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            raise
    
    def merge_with_transcript(self, diarization_json: str, transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge diarization with STT transcript segments.
        Assign speakers to transcript segments based on timestamps.
        
        :param diarization_json: JSON string from diarize().
        :param transcript_segments: List of transcript segments {'text': str, 'start': float, 'end': float, ...}.
        :return: Updated transcript segments with 'speaker' added.
        """
        diarization = json.loads(diarization_json)
        for segment in transcript_segments:
            start, end = segment['start'], segment['end']
            # Find matching speaker (simple overlap check; improve for production)
            for dia in diarization:
                if max(start, dia['start']) < min(end, dia['end']):
                    segment['speaker'] = dia['speaker']
                    break
            else:
                segment['speaker'] = "UNKNOWN"
        
        logger.info("Merged diarization with transcript")
        return transcript_segments

# Example usage (for testing)
if __name__ == "__main__":
    diarizer = SpeakerDiarizer()
    try:
        # Test diarize (replace with actual file)
        dia_json = diarizer.diarize("path/to/sample.wav")
        print(f"Diarization JSON: {dia_json}")
        
        # Test merge (mock transcript)
        mock_segments = [{"text": "Hello", "start": 0.0, "end": 1.0}]
        merged = diarizer.merge_with_transcript(dia_json, mock_segments)
        print(f"Merged: {merged}")
    except Exception as e:
        print(f"Error: {e}")