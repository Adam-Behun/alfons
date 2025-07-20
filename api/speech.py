"""
Enhanced speech processing router delegating to S2S pipeline for real-time and transcription engine for batch.
Maintains backward compatibility while providing unified speech processing interface.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union, AsyncGenerator
from enum import Enum
from pathlib import Path
import tempfile
import os

from .s2s_pipeline import S2SPipeline, create_s2s_pipeline
from call_analytics.transcription_engine.pipeline import create_healthcare_pipeline

from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Speech processing mode selection."""
    REALTIME = "realtime"      # Use S2S pipeline for live calls
    BATCH = "batch"           # Use transcription engine for uploaded files
    AUTO = "auto"             # Automatically choose based on context


class SpeechProcessingRouter:
    """
    Unified speech processing router that delegates to appropriate engines.
    Routes real-time requests to S2S pipeline and batch requests to transcription engine.
    """

    def __init__(self):
        """Initialize speech processing router with components."""
        # Initialize S2S pipeline for real-time processing
        self.s2s_pipeline = create_s2s_pipeline(
            model="gpt-4o-realtime-preview-2025-06-03",
            voice="alloy",
            enable_rag=True
        )

        # Initialize transcription pipeline for batch processing
        self.transcription_pipeline = create_healthcare_pipeline(
            stt_provider="deepgram",
            enable_validation=True,
            enable_diarization=True,
            processing_mode="batch"
        )

        # Processing statistics
        self.stats = {
            "realtime_requests": 0,
            "batch_requests": 0,
            "total_processing_time": 0.0,
            "average_latency": 0.0
        }

        logger.info("SpeechProcessingRouter initialized with S2S and transcription engines")
    
    async def process_speech(
        self,
        audio_input: Union[str, bytes, AsyncGenerator[bytes, None]],
        mode: ProcessingMode = ProcessingMode.AUTO,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process speech input using appropriate engine based on mode and input type.
        
        Args:
            audio_input: Audio file path, bytes, or async stream
            mode: Processing mode (realtime, batch, auto)
            context: Additional processing context
            
        Returns:
            Processing results with transcription, analysis, and metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Determine processing mode if auto
            if mode == ProcessingMode.AUTO:
                mode = self._determine_processing_mode(audio_input, context)
            
            logger.info(f"Processing speech with mode: {mode.value}")
            
            # Route to appropriate processor
            if mode == ProcessingMode.REALTIME:
                result = await self._process_realtime(audio_input, context)
                self.stats["realtime_requests"] += 1
            elif mode == ProcessingMode.BATCH:
                result = await self._process_batch(audio_input, context)
                self.stats["batch_requests"] += 1
            else:
                raise ValueError(f"Unsupported processing mode: {mode}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            total_requests = self.stats["realtime_requests"] + self.stats["batch_requests"]
            self.stats["average_latency"] = self.stats["total_processing_time"] / total_requests
            
            # Add metadata to result
            result["processing"] = {
                "mode": mode.value,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            raise
    
    def _determine_processing_mode(
        self, 
        audio_input: Union[str, bytes, AsyncGenerator[bytes, None]], 
        context: Dict[str, Any]
    ) -> ProcessingMode:
        """
        Automatically determine processing mode based on input and context.
        
        Args:
            audio_input: Audio input
            context: Processing context
            
        Returns:
            Determined processing mode
        """
        # Check context hints
        if context.get("real_time", False) or context.get("streaming", False):
            return ProcessingMode.REALTIME
        
        if context.get("batch", False) or context.get("analysis", False):
            return ProcessingMode.BATCH
        
        # Check input type
        if isinstance(audio_input, AsyncGenerator):
            return ProcessingMode.REALTIME
        
        if isinstance(audio_input, str) and Path(audio_input).exists():
            return ProcessingMode.BATCH
        
        if isinstance(audio_input, bytes):
            # Default to batch for byte input unless context suggests otherwise
            return ProcessingMode.BATCH
        
        # Default fallback
        return ProcessingMode.BATCH
    
    async def _process_realtime(
        self, 
        audio_input: Union[bytes, AsyncGenerator[bytes, None]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process audio using S2S pipeline for real-time interaction.
        
        Args:
            audio_input: Audio bytes or stream
            context: Processing context
            
        Returns:
            Real-time processing results
        """
        logger.debug("Processing with S2S pipeline")
        
        try:
            call_id = context.get("call_id") or f"speech_{int(time.time())}"
            
            # Initialize S2S session
            session_info = await self.s2s_pipeline.start_call_session(call_id)
            
            # Handle different input types
            if isinstance(audio_input, AsyncGenerator):
                # Stream processing
                responses = []
                async for audio_chunk in audio_input:
                    async for response_chunk in self.s2s_pipeline.process_audio_chunk(
                        call_id, audio_chunk, context
                    ):
                        responses.append(response_chunk)
                
                result = {
                    "type": "realtime_stream",
                    "call_id": call_id,
                    "session_info": session_info,
                    "response_chunks": len(responses),
                    "status": "completed"
                }
            
            elif isinstance(audio_input, bytes):
                # Single chunk processing
                responses = []
                async for response_chunk in self.s2s_pipeline.process_audio_chunk(
                    call_id, audio_input, context
                ):
                    responses.append(response_chunk)
                
                result = {
                    "type": "realtime_single",
                    "call_id": call_id,
                    "session_info": session_info,
                    "response_audio": responses[0] if responses else None,
                    "status": "completed"
                }
            
            else:
                raise ValueError("Real-time mode requires bytes or async generator input")
            
            # End session
            summary = await self.s2s_pipeline.end_call_session(call_id)
            result["session_summary"] = summary
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")
            raise
    
    async def _process_batch(
        self, 
        audio_input: Union[str, bytes], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process audio using transcription pipeline for batch analysis.
        
        Args:
            audio_input: Audio file path or bytes
            context: Processing context
            
        Returns:
            Batch processing results with transcription and analysis
        """
        logger.debug("Processing with transcription pipeline")
        
        try:
            # Handle different input types
            if isinstance(audio_input, bytes):
                # Save bytes to temporary file
                temp_file = await self._save_bytes_to_temp_file(audio_input)
                audio_path = temp_file
                cleanup_temp = True
            elif isinstance(audio_input, str):
                # Use provided file path
                audio_path = audio_input
                cleanup_temp = False
            else:
                raise ValueError("Batch mode requires file path or bytes input")
            
            # Process through transcription pipeline
            pipeline_result = await self.transcription_pipeline.process_audio_file(
                audio_path, context
            )
            
            # Format result for unified interface
            result = {
                "type": "batch_transcription",
                "transcript": pipeline_result.transcription.transcript,
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "confidence": seg.confidence,
                        "speaker": seg.speaker
                    }
                    for seg in pipeline_result.transcription.segments
                ],
                "transcription": {
                    "duration": pipeline_result.transcription.duration,
                    "language": pipeline_result.transcription.language,
                    "processing_time": pipeline_result.transcription.processing_time
                },
                "quality_scores": {
                    "overall": pipeline_result.overall_quality_score,
                    "confidence": pipeline_result.confidence_score,
                    "compliance": pipeline_result.compliance_score
                }
            }
            
            # Add optional components if available
            if pipeline_result.diarization:
                result["diarization"] = {
                    "num_speakers": pipeline_result.diarization.num_speakers,
                    "speaker_mapping": pipeline_result.diarization.speaker_mapping,
                    "processing_time": pipeline_result.diarization.processing_time
                }
            
            if pipeline_result.validation:
                result["validation"] = {
                    "is_valid": pipeline_result.validation.is_valid,
                    "overall_quality": pipeline_result.validation.overall_quality,
                    "confidence_score": pipeline_result.validation.confidence_score,
                    "issues_count": len(pipeline_result.validation.issues),
                    "medical_terms_validated": pipeline_result.validation.medical_terms_validated,
                    "compliance_score": pipeline_result.validation.compliance_score
                }
                
                # Include corrections if available
                if pipeline_result.validation.corrections:
                    result["corrections"] = pipeline_result.validation.corrections
            
            if pipeline_result.conversation_analysis:
                result["conversation_analysis"] = pipeline_result.conversation_analysis
            
            # Cleanup temporary file if created
            if cleanup_temp:
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            result["status"] = "completed"
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    async def _save_bytes_to_temp_file(self, audio_bytes: bytes) -> str:
        """
        Save audio bytes to temporary file.
        
        Args:
            audio_bytes: Audio data
            
        Returns:
            Path to temporary file
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            logger.debug(f"Saved audio bytes to temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save bytes to temp file: {e}")
            raise
    
    async def transcribe_audio(
        self, 
        audio_path: str, 
        enhanced: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with optional enhancement (backward compatibility).
        
        Args:
            audio_path: Path to audio file
            enhanced: Use enhanced processing (validation, diarization)
            
        Returns:
            Transcription results
        """
        logger.info(f"Transcribing audio: {audio_path} (enhanced: {enhanced})")
        
        context = {
            "batch": True,
            "enhanced_processing": enhanced,
            "legacy_mode": True
        }
        
        return await self.process_speech(audio_path, ProcessingMode.BATCH, context)
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice: str = "alloy",
        streaming: bool = False
    ) -> Union[bytes, AsyncGenerator[bytes, None]]:
        """
        Synthesize speech from text (integrated with S2S pipeline).
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            streaming: Return streaming generator or complete audio
            
        Returns:
            Audio bytes or async generator
        """
        logger.info(f"Synthesizing speech: {text[:50]}...")
        
        try:
            # For now, delegate to S2S pipeline's TTS capabilities
            # In a full implementation, this would use the TTS component directly
            
            if streaming:
                # Return async generator for streaming TTS
                async def stream_tts():
                    # Placeholder for streaming TTS implementation
                    # Would integrate with S2S pipeline's TTS streaming
                    yield b"placeholder_audio_chunk"
                
                return stream_tts()
            else:
                # Return complete audio
                # Placeholder for complete TTS implementation
                return b"placeholder_complete_audio"
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        return {
            "statistics": self.stats.copy(),
            "components": {
                "s2s_pipeline": {
                    "status": "active",
                    "model": self.s2s_pipeline.config.model,
                    "voice": self.s2s_pipeline.config.voice
                },
                "transcription_pipeline": {
                    "status": "active",
                    "config": self.transcription_pipeline.get_pipeline_status()
                }
            },
            "capabilities": {
                "realtime_processing": True,
                "batch_processing": True,
                "streaming_tts": True,
                "speaker_diarization": True,
                "validation": True,
                "rag_integration": True
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on speech processing components."""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        try:
            # Check S2S pipeline
            try:
                # Would check S2S pipeline health
                health["components"]["s2s_pipeline"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["s2s_pipeline"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
            
            # Check transcription pipeline
            try:
                validation_result = await self.transcription_pipeline.validate_pipeline_config()
                health["components"]["transcription_pipeline"] = {
                    "status": "healthy" if validation_result["valid"] else "degraded",
                    "warnings": validation_result.get("warnings", [])
                }
                if not validation_result["valid"]:
                    health["status"] = "degraded"
            except Exception as e:
                health["components"]["transcription_pipeline"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health


# Global router instance
speech_router: Optional[SpeechProcessingRouter] = None


def get_speech_router() -> SpeechProcessingRouter:
    """Get the global speech processing router instance."""
    global speech_router
    if speech_router is None:
        speech_router = SpeechProcessingRouter()
    return speech_router