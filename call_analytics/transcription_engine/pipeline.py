"""
Unified transcription pipeline orchestrating audio processing, STT, diarization, and validation.
Provides both batch and real-time processing capabilities for healthcare calls.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from enum import Enum

from .stt_engine import STTEngine, TranscriptionResult
from .audio_processor import AudioProcessor, AudioMetrics
from .speaker_diarization import SpeakerDiarizer, DiarizationResult
from .transcript_validator import TranscriptValidator, ValidationResult

from shared.config import config
from shared.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Processing mode for transcription pipeline."""
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"


@dataclass
class PipelineConfig:
    """Configuration for transcription pipeline."""
    # STT Configuration
    stt_provider: str = "deepgram"
    enable_diarization: bool = True
    language: Optional[str] = None
    
    # Audio Processing
    enhance_audio: bool = True
    healthcare_mode: bool = True
    chunk_long_audio: bool = True
    max_chunk_duration: int = 300  # 5 minutes
    
    # Speaker Diarization
    expected_speakers: int = 2
    assign_speaker_roles: bool = True
    
    # Validation
    enable_validation: bool = True
    validation_level: str = "comprehensive"  # "basic", "standard", "comprehensive"
    
    # Performance
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Healthcare Specific
    hipaa_compliance_check: bool = True
    medical_terminology_validation: bool = True
    prior_auth_context: bool = True


@dataclass 
class PipelineResult:
    """Complete pipeline processing result."""
    # Core Results
    transcription: TranscriptionResult
    diarization: Optional[DiarizationResult] = None
    validation: Optional[ValidationResult] = None
    
    # Metadata
    audio_metrics: Optional[AudioMetrics] = None
    processing_time: float = 0.0
    pipeline_config: Optional[PipelineConfig] = None
    
    # Enhanced Data
    enhanced_segments: List[Dict[str, Any]] = None
    conversation_analysis: Optional[Dict[str, Any]] = None
    
    # Quality Metrics
    overall_quality_score: float = 0.0
    confidence_score: float = 0.0
    compliance_score: float = 0.0


class TranscriptionPipeline:
    """Unified transcription pipeline for healthcare audio processing."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize transcription pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.stt_engine = STTEngine(preferred_provider=self.config.stt_provider)
        self.audio_processor = AudioProcessor()
        
        # Optional components based on config
        self.speaker_diarizer = SpeakerDiarizer() if self.config.enable_diarization else None
        self.validator = TranscriptValidator() if self.config.enable_validation else None
        
        logger.info(f"TranscriptionPipeline initialized with mode: {self.config.processing_mode.value}")
    
    async def process_audio_file(
        self,
        audio_path: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process audio file through complete transcription pipeline.
        
        Args:
            audio_path: Path to audio file
            context: Additional context for processing
            
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        logger.info(f"Starting pipeline processing for: {audio_path}")
        
        try:
            # Phase 1: Audio Analysis and Preprocessing
            audio_metrics = await self._analyze_audio(audio_path)
            processed_audio_path = await self._preprocess_audio(audio_path, audio_metrics)
            
            # Phase 2: Determine processing strategy
            chunks = await self._prepare_audio_chunks(processed_audio_path, audio_metrics)
            
            # Phase 3: Parallel processing if enabled and beneficial
            if self.config.parallel_processing and len(chunks) > 1:
                results = await self._process_chunks_parallel(chunks, context)
                transcription = await self._merge_chunk_results(results)
            else:
                # Single file processing
                transcription = await self._process_single_audio(processed_audio_path, context)
            
            # Phase 4: Speaker Diarization (if enabled)
            diarization = None
            if self.speaker_diarizer:
                diarization = await self._perform_diarization(processed_audio_path, context)
                
                # Merge transcription with diarization
                transcription = await self._merge_transcription_diarization(transcription, diarization)
            
            # Phase 5: Validation (if enabled)
            validation = None
            if self.validator:
                validation = await self._validate_transcript(transcription, context)
            
            # Phase 6: Generate enhanced results
            enhanced_segments = await self._enhance_segments(transcription, diarization, validation)
            conversation_analysis = await self._analyze_conversation(enhanced_segments, diarization)
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(transcription, validation, diarization)
            
            # Create final result
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                transcription=transcription,
                diarization=diarization,
                validation=validation,
                audio_metrics=audio_metrics,
                processing_time=processing_time,
                pipeline_config=self.config,
                enhanced_segments=enhanced_segments,
                conversation_analysis=conversation_analysis,
                overall_quality_score=quality_scores["overall"],
                confidence_score=quality_scores["confidence"],
                compliance_score=quality_scores["compliance"]
            )
            
            # Cleanup temporary files
            await self._cleanup_processing_files([processed_audio_path] + chunks)
            
            logger.info(f"Pipeline processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio stream for real-time transcription.
        
        Args:
            audio_stream: Async generator yielding audio chunks
            context: Processing context
            
        Yields:
            Partial transcription results as they become available
        """
        if not self.stt_engine.supports_streaming():
            raise NotImplementedError("Current STT provider doesn't support streaming")
        
        logger.info("Starting real-time stream processing")
        
        try:
            async for segment in self.stt_engine.transcribe_stream(audio_stream):
                # Real-time processing yields partial results
                partial_result = {
                    "segment": asdict(segment),
                    "timestamp": time.time(),
                    "is_partial": True
                }
                yield partial_result
                
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise
    
    async def _analyze_audio(self, audio_path: str) -> AudioMetrics:
        """Analyze audio file quality and characteristics."""
        logger.debug("Analyzing audio quality")
        return await self.audio_processor.analyze_audio_quality(audio_path)
    
    async def _preprocess_audio(self, audio_path: str, metrics: AudioMetrics) -> str:
        """Preprocess audio for optimal transcription."""
        if not self.config.enhance_audio:
            return audio_path
        
        logger.debug("Preprocessing audio")
        return await self.audio_processor.enhance_audio(
            audio_path,
            healthcare_mode=self.config.healthcare_mode
        )
    
    async def _prepare_audio_chunks(self, audio_path: str, metrics: AudioMetrics) -> List[str]:
        """Prepare audio chunks if needed."""
        if not self.config.chunk_long_audio or metrics.duration <= self.config.max_chunk_duration:
            return [audio_path]
        
        logger.debug(f"Chunking audio ({metrics.duration}s > {self.config.max_chunk_duration}s)")
        return await self.audio_processor.chunk_audio(
            audio_path,
            chunk_duration=self.config.max_chunk_duration
        )
    
    async def _process_chunks_parallel(
        self,
        chunks: List[str],
        context: Optional[Dict[str, Any]]
    ) -> List[TranscriptionResult]:
        """Process audio chunks in parallel."""
        logger.debug(f"Processing {len(chunks)} chunks in parallel")
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_chunk(chunk_path: str) -> TranscriptionResult:
            async with semaphore:
                return await self.stt_engine.transcribe_batch(
                    chunk_path,
                    diarize=False,  # Diarization done separately on full audio
                    language=self.config.language
                )
        
        tasks = [process_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
    
    async def _process_single_audio(
        self,
        audio_path: str,
        context: Optional[Dict[str, Any]]
    ) -> TranscriptionResult:
        """Process single audio file."""
        logger.debug("Processing single audio file")
        
        return await self.stt_engine.transcribe_batch(
            audio_path,
            diarize=self.config.enable_diarization and not self.speaker_diarizer,  # Use STT diarization if no separate diarizer
            language=self.config.language
        )
    
    async def _merge_chunk_results(self, results: List[TranscriptionResult]) -> TranscriptionResult:
        """Merge results from multiple chunks."""
        logger.debug("Merging chunk results")
        
        # Combine transcripts
        full_transcript = " ".join(result.transcript for result in results)
        
        # Merge segments with time offset adjustment
        all_segments = []
        time_offset = 0.0
        
        for result in results:
            for segment in result.segments:
                adjusted_segment = TranscriptionSegment(
                    text=segment.text,
                    start=segment.start + time_offset,
                    end=segment.end + time_offset,
                    confidence=segment.confidence,
                    speaker=segment.speaker
                )
                all_segments.append(adjusted_segment)
            
            # Update offset for next chunk
            if result.segments:
                time_offset = max(seg.end + time_offset for seg in result.segments)
        
        # Calculate combined metrics
        total_duration = sum(result.duration for result in results)
        avg_processing_time = sum(result.processing_time for result in results) / len(results)
        
        return TranscriptionResult(
            transcript=full_transcript,
            segments=all_segments,
            duration=total_duration,
            language=results[0].language if results else None,
            processing_time=avg_processing_time
        )
    
    async def _perform_diarization(
        self,
        audio_path: str,
        context: Optional[Dict[str, Any]]
    ) -> DiarizationResult:
        """Perform speaker diarization."""
        logger.debug("Performing speaker diarization")
        
        return await self.speaker_diarizer.diarize_healthcare_call(
            audio_path,
            expected_speakers=self.config.expected_speakers,
            context=context
        )
    
    async def _merge_transcription_diarization(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult
    ) -> TranscriptionResult:
        """Merge transcription with diarization results."""
        logger.debug("Merging transcription with diarization")
        
        # Convert segments to dict format for merging
        transcript_segments = [asdict(seg) for seg in transcription.segments]
        
        # Merge using diarizer
        enhanced_segments = await self.speaker_diarizer.merge_with_transcript(
            diarization, transcript_segments
        )
        
        # Convert back to TranscriptionSegment objects
        merged_segments = []
        for seg in enhanced_segments:
            merged_segments.append(TranscriptionSegment(
                text=seg['text'],
                start=seg['start'],
                end=seg['end'],
                confidence=seg['confidence'],
                speaker=seg.get('speaker')
            ))
        
        # Return updated transcription result
        return TranscriptionResult(
            transcript=transcription.transcript,
            segments=merged_segments,
            duration=transcription.duration,
            language=transcription.language,
            processing_time=transcription.processing_time
        )
    
    async def _validate_transcript(
        self,
        transcription: TranscriptionResult,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate transcription quality and compliance."""
        logger.debug("Validating transcript")
        
        # Convert segments to dict format for validation
        segments_dict = [asdict(seg) for seg in transcription.segments]
        
        return await self.validator.validate_healthcare_transcript(
            transcription.transcript,
            segments_dict,
            context
        )
    
    async def _enhance_segments(
        self,
        transcription: TranscriptionResult,
        diarization: Optional[DiarizationResult],
        validation: Optional[ValidationResult]
    ) -> List[Dict[str, Any]]:
        """Create enhanced segments with all available metadata."""
        logger.debug("Enhancing segments with metadata")
        
        enhanced = []
        for segment in transcription.segments:
            enhanced_segment = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "confidence": segment.confidence,
                "speaker": segment.speaker,
                "duration": segment.end - segment.start
            }
            
            # Add validation metadata if available
            if validation and validation.corrections:
                corrected_text = enhanced_segment["text"]
                for original, correction in validation.corrections.items():
                    corrected_text = corrected_text.replace(original, correction)
                if corrected_text != enhanced_segment["text"]:
                    enhanced_segment["corrected_text"] = corrected_text
            
            enhanced.append(enhanced_segment)
        
        return enhanced
    
    async def _analyze_conversation(
        self,
        segments: List[Dict[str, Any]],
        diarization: Optional[DiarizationResult]
    ) -> Dict[str, Any]:
        """Analyze conversation patterns and flow."""
        if not diarization:
            return {}
        
        logger.debug("Analyzing conversation patterns")
        
        return await self.speaker_diarizer.analyze_conversation_flow(diarization)
    
    def _calculate_quality_scores(
        self,
        transcription: TranscriptionResult,
        validation: Optional[ValidationResult],
        diarization: Optional[DiarizationResult]
    ) -> Dict[str, float]:
        """Calculate overall quality scores."""
        
        # Base confidence from transcription
        if transcription.segments:
            base_confidence = sum(seg.confidence for seg in transcription.segments) / len(transcription.segments)
        else:
            base_confidence = 0.0
        
        # Adjust based on validation
        validation_confidence = validation.confidence_score if validation else 1.0
        compliance_score = validation.compliance_score if validation else 1.0
        
        # Calculate overall quality
        overall_quality = (base_confidence + validation_confidence) / 2
        
        return {
            "overall": overall_quality,
            "confidence": validation_confidence,
            "compliance": compliance_score
        }
    
    async def _cleanup_processing_files(self, file_paths: List[str]):
        """Clean up temporary processing files."""
        await self.audio_processor.cleanup_temp_files(file_paths)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline configuration and status."""
        return {
            "config": asdict(self.config),
            "components": {
                "stt_engine": type(self.stt_engine).__name__,
                "audio_processor": type(self.audio_processor).__name__,
                "speaker_diarizer": type(self.speaker_diarizer).__name__ if self.speaker_diarizer else None,
                "validator": type(self.validator).__name__ if self.validator else None
            },
            "capabilities": {
                "streaming_supported": self.stt_engine.supports_streaming(),
                "diarization_enabled": self.config.enable_diarization,
                "validation_enabled": self.config.enable_validation,
                "parallel_processing": self.config.parallel_processing
            }
        }
    
    async def validate_pipeline_config(self) -> Dict[str, Any]:
        """Validate pipeline configuration and component availability."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check STT provider availability
        try:
            # Test with a minimal request (would need actual test)
            logger.debug("Validating STT provider configuration")
        except Exception as e:
            validation_results["errors"].append(f"STT provider unavailable: {e}")
            validation_results["valid"] = False
        
        # Check diarization setup
        if self.config.enable_diarization and not self.speaker_diarizer:
            validation_results["warnings"].append("Diarization enabled but no diarizer available")
        
        # Check validation setup
        if self.config.enable_validation and not self.validator:
            validation_results["warnings"].append("Validation enabled but no validator available")
        
        # Check streaming compatibility
        if self.config.processing_mode == ProcessingMode.STREAMING:
            if not self.stt_engine.supports_streaming():
                validation_results["errors"].append("Streaming mode requested but STT provider doesn't support streaming")
                validation_results["valid"] = False
        
        return validation_results
    
    async def process_batch_files(
        self,
        file_paths: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PipelineResult]:
        """
        Process multiple audio files in batch.
        
        Args:
            file_paths: List of audio file paths
            context: Shared context for all files
            
        Returns:
            List of pipeline results for each file
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        if self.config.parallel_processing:
            # Process files in parallel with semaphore control
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def process_file(file_path: str) -> PipelineResult:
                async with semaphore:
                    return await self.process_audio_file(file_path, context)
            
            tasks = [process_file(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {file_paths[i]}: {result}")
                    # Create error result
                    error_result = PipelineResult(
                        transcription=TranscriptionResult(
                            transcript=f"Error: {str(result)}",
                            segments=[],
                            duration=0.0
                        ),
                        processing_time=0.0,
                        overall_quality_score=0.0
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Sequential processing
            results = []
            for file_path in file_paths:
                try:
                    result = await self.process_audio_file(file_path, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    # Create error result
                    error_result = PipelineResult(
                        transcription=TranscriptionResult(
                            transcript=f"Error: {str(e)}",
                            segments=[],
                            duration=0.0
                        ),
                        processing_time=0.0,
                        overall_quality_score=0.0
                    )
                    results.append(error_result)
            
            return results
    
    def export_pipeline_results(
        self,
        result: PipelineResult,
        format: str = "json",
        include_metadata: bool = True
    ) -> str:
        """
        Export pipeline results in specified format.
        
        Args:
            result: Pipeline result to export
            format: Export format ("json", "txt", "csv")
            include_metadata: Include processing metadata
            
        Returns:
            Formatted export string
        """
        if format == "json":
            return self._export_json_results(result, include_metadata)
        elif format == "txt":
            return self._export_text_results(result, include_metadata)
        elif format == "csv":
            return self._export_csv_results(result)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json_results(self, result: PipelineResult, include_metadata: bool) -> str:
        """Export results as JSON."""
        export_data = {
            "transcript": result.transcription.transcript,
            "segments": [asdict(seg) for seg in result.transcription.segments]
        }
        
        if include_metadata:
            export_data.update({
                "processing_time": result.processing_time,
                "quality_scores": {
                    "overall": result.overall_quality_score,
                    "confidence": result.confidence_score,
                    "compliance": result.compliance_score
                },
                "audio_metrics": asdict(result.audio_metrics) if result.audio_metrics else None,
                "diarization": {
                    "num_speakers": result.diarization.num_speakers,
                    "speaker_mapping": result.diarization.speaker_mapping
                } if result.diarization else None,
                "validation": {
                    "is_valid": result.validation.is_valid,
                    "overall_quality": result.validation.overall_quality,
                    "issues_count": len(result.validation.issues)
                } if result.validation else None,
                "conversation_analysis": result.conversation_analysis,
                "pipeline_config": asdict(result.pipeline_config) if result.pipeline_config else None
            })
        
        return json.dumps(export_data, indent=2)
    
    def _export_text_results(self, result: PipelineResult, include_metadata: bool) -> str:
        """Export results as human-readable text."""
        lines = [
            "TRANSCRIPTION RESULTS",
            "=" * 50,
            "",
            "TRANSCRIPT:",
            "-" * 20,
            result.transcription.transcript,
            "",
            "SEGMENTS:",
            "-" * 20
        ]
        
        for i, segment in enumerate(result.transcription.segments, 1):
            speaker_info = f" [{segment.speaker}]" if segment.speaker else ""
            lines.append(f"{i:3d}. [{segment.start:6.2f}s - {segment.end:6.2f}s]{speaker_info} {segment.text}")
        
        if include_metadata:
            lines.extend([
                "",
                "METADATA:",
                "-" * 20,
                f"Processing Time: {result.processing_time:.2f}s",
                f"Overall Quality: {result.overall_quality_score:.2f}",
                f"Confidence Score: {result.confidence_score:.2f}",
                f"Compliance Score: {result.compliance_score:.2f}"
            ])
            
            if result.diarization:
                lines.extend([
                    f"Speakers Detected: {result.diarization.num_speakers}",
                    f"Speaker Mapping: {result.diarization.speaker_mapping}"
                ])
            
            if result.validation:
                lines.extend([
                    f"Validation Status: {'PASSED' if result.validation.is_valid else 'FAILED'}",
                    f"Validation Quality: {result.validation.overall_quality.upper()}",
                    f"Issues Found: {len(result.validation.issues)}"
                ])
        
        return "\n".join(lines)
    
    def _export_csv_results(self, result: PipelineResult) -> str:
        """Export segments as CSV."""
        lines = ["Start,End,Duration,Speaker,Confidence,Text"]
        
        for segment in result.transcription.segments:
            duration = segment.end - segment.start
            line = f"{segment.start:.2f},{segment.end:.2f},{duration:.2f},{segment.speaker or 'Unknown'},{segment.confidence:.3f},\"{segment.text}\""
            lines.append(line)
        
        return "\n".join(lines)


# Factory function for easy pipeline creation
def create_healthcare_pipeline(
    stt_provider: str = "deepgram",
    enable_validation: bool = True,
    enable_diarization: bool = True,
    processing_mode: str = "batch"
) -> TranscriptionPipeline:
    """
    Factory function to create healthcare-optimized transcription pipeline.
    
    Args:
        stt_provider: STT provider to use
        enable_validation: Enable transcript validation
        enable_diarization: Enable speaker diarization
        processing_mode: Processing mode ("batch", "streaming", "hybrid")
        
    Returns:
        Configured TranscriptionPipeline
    """
    config = PipelineConfig(
        stt_provider=stt_provider,
        enable_diarization=enable_diarization,
        enable_validation=enable_validation,
        processing_mode=ProcessingMode(processing_mode),
        healthcare_mode=True,
        hipaa_compliance_check=True,
        medical_terminology_validation=True,
        prior_auth_context=True
    )
    
    return TranscriptionPipeline(config)