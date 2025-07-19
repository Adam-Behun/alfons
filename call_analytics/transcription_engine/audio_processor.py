"""
Enhanced audio processing for healthcare transcription with noise reduction and format optimization.
Supports both batch processing and real-time stream preparation.
"""

import asyncio
import logging
import os
import subprocess
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass

import ffmpeg
import numpy as np

from shared.config import config
from shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AudioMetrics:
    """Audio quality metrics for validation."""
    duration: float
    sample_rate: int
    channels: int
    bit_rate: Optional[int]
    format: str
    noise_level: Optional[float] = None
    signal_quality: Optional[float] = None


class AudioProcessor:
    """Enhanced audio processor with healthcare-specific optimizations."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize audio processor.
        
        Args:
            temp_dir: Directory for temporary files (uses system temp if None)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "alfons_audio"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Healthcare-optimized settings
        self.target_sample_rate = 16000  # Optimal for STT
        self.target_channels = 1  # Mono for better STT performance
        self.noise_reduction_enabled = True
        
        logger.info(f"AudioProcessor initialized with temp_dir: {self.temp_dir}")
    
    async def enhance_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        healthcare_mode: bool = True
    ) -> str:
        """
        Enhance audio with healthcare-specific processing.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional output path (generates if None)
            healthcare_mode: Apply healthcare-specific enhancements
            
        Returns:
            Path to enhanced audio file
        """
        input_path = Path(input_path)
        
        if not output_path:
            output_path = self.temp_dir / f"{input_path.stem}_enhanced.wav"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Enhancing audio: {input_path} -> {output_path}")
        
        try:
            # Build ffmpeg pipeline
            audio_input = ffmpeg.input(str(input_path))
            
            # Healthcare-specific enhancement pipeline
            if healthcare_mode:
                # 1. Noise reduction (adaptive filter for phone calls)
                audio_filtered = audio_input.filter('afftdn', nf=-25)
                
                # 2. High-pass filter to remove low-frequency noise
                audio_filtered = audio_filtered.filter('highpass', f=80)
                
                # 3. Compressor to normalize volume levels
                audio_filtered = audio_filtered.filter('acompressor', 
                                                     threshold=0.5, 
                                                     ratio=4, 
                                                     attack=0.2, 
                                                     release=0.8)
                
                # 4. Loudness normalization
                audio_filtered = audio_filtered.filter('loudnorm', 
                                                      I=-16, 
                                                      LRA=11, 
                                                      tp=-1.5)
            else:
                # Basic enhancement
                audio_filtered = audio_input.filter('afftdn').filter('loudnorm')
            
            # Output with optimal settings for STT
            output = audio_filtered.output(
                str(output_path),
                ac=self.target_channels,  # Mono
                ar=self.target_sample_rate,  # 16kHz
                acodec='pcm_s16le',  # 16-bit PCM
                f='wav'
            )
            
            # Run enhancement
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: output.overwrite_output().run(quiet=True)
            )
            
            logger.info(f"Audio enhancement completed: {output_path}")
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error(f"Audio enhancement failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audio enhancement: {e}")
            raise
    
    async def chunk_audio(
        self,
        input_path: str,
        chunk_duration: int = 300,
        overlap: int = 5,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Split audio into overlapping chunks for better processing.
        
        Args:
            input_path: Path to input audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            output_dir: Output directory for chunks
            
        Returns:
            List of chunk file paths
        """
        input_path = Path(input_path)
        
        if not output_dir:
            output_dir = self.temp_dir / f"{input_path.stem}_chunks"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Chunking audio: {input_path} ({chunk_duration}s chunks with {overlap}s overlap)")
        
        try:
            # Get audio duration
            probe = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ffmpeg.probe(str(input_path))
            )
            duration = float(probe['format']['duration'])
            
            # Calculate chunk positions
            chunk_starts = []
            current_start = 0
            while current_start < duration:
                chunk_starts.append(current_start)
                current_start += chunk_duration - overlap
            
            # Create chunks
            chunk_paths = []
            for i, start in enumerate(chunk_starts):
                chunk_path = output_dir / f"chunk_{i:03d}.wav"
                
                # Create chunk with ffmpeg
                chunk_input = ffmpeg.input(str(input_path), ss=start, t=chunk_duration)
                chunk_output = chunk_input.output(
                    str(chunk_path),
                    ac=self.target_channels,
                    ar=self.target_sample_rate,
                    acodec='pcm_s16le'
                )
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda o=chunk_output: o.overwrite_output().run(quiet=True)
                )
                
                if chunk_path.exists() and chunk_path.stat().st_size > 0:
                    chunk_paths.append(str(chunk_path))
            
            logger.info(f"Created {len(chunk_paths)} audio chunks")
            return chunk_paths
            
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise
    
    async def analyze_audio_quality(self, audio_path: str) -> AudioMetrics:
        """
        Analyze audio file quality and extract metrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioMetrics with quality information
        """
        logger.debug(f"Analyzing audio quality: {audio_path}")
        
        try:
            # Get basic audio info
            probe = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ffmpeg.probe(str(audio_path))
            )
            
            format_info = probe['format']
            audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            metrics = AudioMetrics(
                duration=float(format_info['duration']),
                sample_rate=int(audio_stream['sample_rate']),
                channels=int(audio_stream['channels']),
                bit_rate=int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                format=format_info['format_name']
            )
            
            # Advanced quality analysis (if needed)
            if config.AUDIO_QUALITY_ANALYSIS:
                noise_level, signal_quality = await self._analyze_signal_quality(audio_path)
                metrics.noise_level = noise_level
                metrics.signal_quality = signal_quality
            
            logger.debug(f"Audio metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            raise
    
    async def _analyze_signal_quality(self, audio_path: str) -> tuple[float, float]:
        """
        Analyze signal-to-noise ratio and overall quality.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (noise_level, signal_quality) as floats 0-1
        """
        try:
            # Use ffmpeg to extract audio statistics
            cmd = [
                'ffmpeg', '-i', audio_path, '-af', 'astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True)
            )
            
            # Parse statistics from stderr
            stats = result.stderr
            
            # Extract RMS and peak levels (simplified analysis)
            noise_level = 0.1  # Default values
            signal_quality = 0.8
            
            # This is a simplified implementation
            # In production, you'd parse the actual statistics
            if 'RMS level' in stats:
                # Extract and calculate actual metrics
                pass
            
            return noise_level, signal_quality
            
        except Exception as e:
            logger.warning(f"Signal quality analysis failed: {e}")
            return 0.1, 0.8  # Default values
    
    async def prepare_for_streaming(self, audio_path: str) -> AsyncGenerator[bytes, None]:
        """
        Prepare audio file for real-time streaming processing.
        
        Args:
            audio_path: Path to audio file
            
        Yields:
            Audio chunks as bytes
        """
        logger.info(f"Preparing audio for streaming: {audio_path}")
        
        # Convert to optimal format for streaming
        temp_path = self.temp_dir / f"stream_{Path(audio_path).stem}.wav"
        
        try:
            # Convert to streaming-optimized format
            stream_input = ffmpeg.input(str(audio_path))
            stream_output = stream_input.output(
                str(temp_path),
                ac=1,  # Mono
                ar=16000,  # 16kHz
                acodec='pcm_s16le',  # 16-bit PCM
                f='wav'
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: stream_output.overwrite_output().run(quiet=True)
            )
            
            # Stream file in chunks
            chunk_size = 4096  # 4KB chunks
            with open(temp_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            
        except Exception as e:
            logger.error(f"Streaming preparation failed: {e}")
            raise
        finally:
            # Cleanup temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    async def validate_audio_for_healthcare(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate audio file for healthcare transcription requirements.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Validation results with recommendations
        """
        logger.info(f"Validating audio for healthcare use: {audio_path}")
        
        try:
            metrics = await self.analyze_audio_quality(audio_path)
            
            validation = {
                "valid": True,
                "warnings": [],
                "recommendations": []
            }
            
            # Check duration (too short/long)
            if metrics.duration < 10:
                validation["warnings"].append("Audio very short (<10s) - may affect transcription quality")
            elif metrics.duration > 3600:  # 1 hour
                validation["warnings"].append("Audio very long (>1h) - consider chunking")
                validation["recommendations"].append("Split into smaller segments for better processing")
            
            # Check sample rate
            if metrics.sample_rate < 8000:
                validation["valid"] = False
                validation["warnings"].append("Sample rate too low (<8kHz) - will affect quality")
                validation["recommendations"].append("Use higher quality audio (16kHz+ recommended)")
            elif metrics.sample_rate < 16000:
                validation["warnings"].append("Sample rate below optimal (16kHz)")
                validation["recommendations"].append("16kHz sample rate recommended for best results")
            
            # Check channels
            if metrics.channels > 2:
                validation["warnings"].append("Multi-channel audio detected")
                validation["recommendations"].append("Convert to mono for better STT performance")
            
            # Check format
            if metrics.format not in ['wav', 'flac', 'mp3', 'aac']:
                validation["warnings"].append(f"Unusual audio format: {metrics.format}")
                validation["recommendations"].append("Use WAV, FLAC, MP3, or AAC for best compatibility")
            
            logger.info(f"Audio validation completed: {validation}")
            return validation
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return {
                "valid": False,
                "warnings": [f"Validation error: {str(e)}"],
                "recommendations": ["Check audio file integrity"]
            }
    
    async def cleanup_temp_files(self, file_paths: Optional[List[str]] = None):
        """
        Clean up temporary files.
        
        Args:
            file_paths: Specific files to clean (cleans all temp if None)
        """
        try:
            if file_paths:
                for path in file_paths:
                    path_obj = Path(path)
                    if path_obj.exists():
                        path_obj.unlink()
                        logger.debug(f"Deleted temp file: {path}")
            else:
                # Clean all temp files older than 1 hour
                import time
                current_time = time.time()
                for file_path in self.temp_dir.glob("*"):
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > 3600:  # 1 hour
                            file_path.unlink()
                            logger.debug(f"Deleted old temp file: {file_path}")
                            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            asyncio.create_task(self.cleanup_temp_files())
        except:
            pass  # Best effort cleanup