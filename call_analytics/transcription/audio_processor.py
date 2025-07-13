import logging
import subprocess
import os
from typing import Optional, List
import ffmpeg  # Assuming ffmpeg-python is installed

from config.settings import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio file processing: enhancement, noise reduction, and chunking.
    Uses ffmpeg for operations like noise reduction and splitting into chunks.
    Supports MP3/WAV inputs, outputs processed WAV for STT compatibility.
    Ensures error handling and logging.
    """
    
    def __init__(self, temp_dir: str = "./temp_audio"):
        """
        Initialize the audio processor with a temporary directory for processed files.
        
        :param temp_dir: Directory for temporary processed audio files.
        """
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"AudioProcessor initialized with temp_dir: {self.temp_dir}")
    
    def enhance_audio(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Enhance audio: noise reduction, normalization, convert to WAV.
        
        :param input_path: Path to the input audio file (MP3/WAV).
        :param output_path: Optional output path; defaults to temp_dir.
        :return: Path to the enhanced audio file.
        """
        if not output_path:
            base_name = os.path.basename(input_path).rsplit('.', 1)[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_enhanced.wav")
        
        try:
            # Use ffmpeg for noise reduction (afftdn) and normalization (loudnorm)
            (
                ffmpeg
                .input(input_path)
                .filter('afftdn')  # Noise reduction
                .filter('loudnorm')  # Normalization
                .output(output_path, ac=1, ar='16k')  # Mono, 16kHz for STT
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Enhanced audio saved to: {output_path}")
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Error enhancing audio: {e}")
            raise
    
    def chunk_audio(self, input_path: str, chunk_duration: int = 300, output_dir: Optional[str] = None) -> List[str]:
        """
        Split audio into chunks for easier processing (e.g., long calls).
        
        :param input_path: Path to the input audio file.
        :param chunk_duration: Duration of each chunk in seconds (default 5 min).
        :param output_dir: Optional directory for chunks; defaults to temp_dir.
        :return: List of paths to chunk files.
        """
        if not output_dir:
            output_dir = self.temp_dir
        
        base_name = os.path.basename(input_path).rsplit('.', 1)[0]
        chunk_pattern = os.path.join(output_dir, f"{base_name}_chunk_%03d.wav")
        
        try:
            # Get audio duration
            probe = ffmpeg.probe(input_path)
            duration = float(probe['format']['duration'])
            
            # Split into chunks
            subprocess.run([
                'ffmpeg', '-i', input_path, '-f', 'segment', '-segment_time', str(chunk_duration),
                '-c', 'copy', chunk_pattern
            ], check=True, capture_output=True)
            
            chunks = [os.path.join(output_dir, f"{base_name}_chunk_{i:03d}.wav") for i in range(int(duration / chunk_duration) + 1)]
            chunks = [c for c in chunks if os.path.exists(c)]
            
            logger.info(f"Split audio into {len(chunks)} chunks")
            return chunks
        except subprocess.CalledProcessError as e:
            logger.error(f"Error chunking audio: {e}")
            raise
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg probe error: {e}")
            raise
    
    def clean_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary files.
        
        :param file_paths: List of file paths to delete.
        """
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Deleted temp file: {path}")

# Example usage (for testing)
if __name__ == "__main__":
    processor = AudioProcessor()
    try:
        # Test enhance (replace with actual file)
        enhanced = processor.enhance_audio("path/to/sample.mp3")
        print(f"Enhanced: {enhanced}")
        
        # Test chunk
        chunks = processor.chunk_audio(enhanced)
        print(f"Chunks: {chunks}")
        
        # Clean
        processor.clean_temp_files(chunks + [enhanced])
    except Exception as e:
        print(f"Error: {e}")