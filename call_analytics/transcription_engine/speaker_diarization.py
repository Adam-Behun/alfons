"""
Enhanced speaker diarization for healthcare calls with provider integration.
Optimized for phone calls between healthcare providers and insurance representatives.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from abc import ABC, abstractmethod

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

from shared.config import config
from shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    """Speaker segment with metadata."""
    speaker_id: str
    start: float
    end: float
    confidence: float
    speaker_role: Optional[str] = None  # 'provider', 'insurance', 'unknown'


@dataclass
class DiarizationResult:
    """Complete diarization result."""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_mapping: Dict[str, str]
    processing_time: float


class ISpeakerDiarizationProvider(ABC):
    """Abstract interface for speaker diarization providers."""
    
    @abstractmethod
    async def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """Perform speaker diarization on audio file."""
        pass
    
    @abstractmethod
    def supports_realtime(self) -> bool:
        """Check if provider supports real-time diarization."""
        pass


class PyAnnoteProvider(ISpeakerDiarizationProvider):
    """PyAnnote.audio provider for speaker diarization."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", auth_token: Optional[str] = None):
        """
        Initialize PyAnnote diarization pipeline.
        
        Args:
            model_name: HuggingFace model name
            auth_token: HuggingFace authentication token
        """
        self.model_name = model_name
        self.auth_token = auth_token or config.HUGGINGFACE_TOKEN
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline."""
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name, 
                use_auth_token=self.auth_token
            )
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Using GPU for diarization")
            else:
                logger.info("Using CPU for diarization")
                
            logger.info(f"PyAnnote pipeline initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyAnnote pipeline: {e}")
            raise
    
    async def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization using PyAnnote.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers hint
            
        Returns:
            DiarizationResult with speaker segments
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting diarization for: {audio_path}")
        
        try:
            # Run diarization in executor to avoid blocking
            diarization = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_diarization,
                audio_path,
                num_speakers
            )
            
            # Convert to standard format
            segments = []
            speaker_mapping = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map internal speaker IDs to standardized format
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"SPEAKER_{len(speaker_mapping) + 1}"
                
                segments.append(SpeakerSegment(
                    speaker_id=speaker_mapping[speaker],
                    start=turn.start,
                    end=turn.end,
                    confidence=1.0,  # PyAnnote doesn't provide confidence scores
                    speaker_role=None  # Will be inferred later
                ))
            
            processing_time = time.time() - start_time
            
            result = DiarizationResult(
                segments=segments,
                num_speakers=len(speaker_mapping),
                speaker_mapping=speaker_mapping,
                processing_time=processing_time
            )
            
            logger.info(f"Diarization completed in {processing_time:.2f}s - {len(segments)} segments, {len(speaker_mapping)} speakers")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def _run_diarization(self, audio_path: str, num_speakers: Optional[int]) -> Annotation:
        """Run diarization pipeline (blocking operation)."""
        kwargs = {}
        if num_speakers:
            kwargs['num_speakers'] = num_speakers
        
        return self.pipeline(audio_path, **kwargs)
    
    def supports_realtime(self) -> bool:
        """PyAnnote doesn't support real-time diarization."""
        return False


class SpeakerDiarizer:
    """Enhanced speaker diarization with healthcare-specific optimizations."""
    
    def __init__(self, provider: str = "pyannote"):
        """
        Initialize speaker diarizer.
        
        Args:
            provider: Diarization provider ('pyannote')
        """
        self.provider = self._initialize_provider(provider)
        self.healthcare_role_mapping = {
            "provider": ["doctor", "physician", "nurse", "provider", "clinic"],
            "insurance": ["insurance", "representative", "agent", "payer", "plan"]
        }
        logger.info(f"SpeakerDiarizer initialized with {type(self.provider).__name__}")
    
    def _initialize_provider(self, provider_name: str) -> ISpeakerDiarizationProvider:
        """Initialize diarization provider."""
        if provider_name == "pyannote":
            return PyAnnoteProvider()
        else:
            raise ValueError(f"Unsupported diarization provider: {provider_name}")
    
    async def diarize_healthcare_call(
        self,
        audio_path: str,
        expected_speakers: int = 2,
        context: Optional[Dict[str, Any]] = None
    ) -> DiarizationResult:
        """
        Perform diarization optimized for healthcare calls.
        
        Args:
            audio_path: Path to audio file
            expected_speakers: Expected number of speakers (default 2 for provider-insurance calls)
            context: Additional context about the call
            
        Returns:
            DiarizationResult with healthcare-specific speaker role assignments
        """
        logger.info(f"Starting healthcare call diarization: {audio_path}")
        
        try:
            # Perform base diarization
            result = await self.provider.diarize(audio_path, expected_speakers)
            
            # Apply healthcare-specific post-processing
            enhanced_result = await self._enhance_with_healthcare_context(result, context)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Healthcare diarization failed: {e}")
            raise
    
    async def _enhance_with_healthcare_context(
        self,
        result: DiarizationResult,
        context: Optional[Dict[str, Any]]
    ) -> DiarizationResult:
        """
        Enhance diarization with healthcare-specific role assignment.
        
        Args:
            result: Base diarization result
            context: Call context information
            
        Returns:
            Enhanced DiarizationResult with role assignments
        """
        # Assign speaker roles based on patterns
        role_assignments = await self._infer_speaker_roles(result.segments, context)
        
        # Update segments with role assignments
        enhanced_segments = []
        for segment in result.segments:
            enhanced_segment = SpeakerSegment(
                speaker_id=segment.speaker_id,
                start=segment.start,
                end=segment.end,
                confidence=segment.confidence,
                speaker_role=role_assignments.get(segment.speaker_id, "unknown")
            )
            enhanced_segments.append(enhanced_segment)
        
        return DiarizationResult(
            segments=enhanced_segments,
            num_speakers=result.num_speakers,
            speaker_mapping=result.speaker_mapping,
            processing_time=result.processing_time
        )
    
    async def _infer_speaker_roles(
        self,
        segments: List[SpeakerSegment],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Infer speaker roles based on patterns and context.
        
        Args:
            segments: Speaker segments
            context: Call context
            
        Returns:
            Mapping of speaker_id to role
        """
        role_assignments = {}
        
        # Simple heuristic: first speaker is often the caller (provider)
        # Second speaker is often the receiver (insurance)
        unique_speakers = list(set(segment.speaker_id for segment in segments))
        
        if len(unique_speakers) == 2:
            # Typical provider-insurance call
            role_assignments[unique_speakers[0]] = "provider"
            role_assignments[unique_speakers[1]] = "insurance"
        elif len(unique_speakers) > 2:
            # Multi-party call - assign based on speaking patterns
            speaker_talk_time = {}
            for segment in segments:
                duration = segment.end - segment.start
                speaker_talk_time[segment.speaker_id] = speaker_talk_time.get(segment.speaker_id, 0) + duration
            
            # Sort by talk time
            sorted_speakers = sorted(speaker_talk_time.items(), key=lambda x: x[1], reverse=True)
            
            # Assign roles based on talk time and context
            for i, (speaker_id, _) in enumerate(sorted_speakers):
                if i == 0:
                    role_assignments[speaker_id] = "provider"
                elif i == 1:
                    role_assignments[speaker_id] = "insurance"
                else:
                    role_assignments[speaker_id] = "other"
        else:
            # Single speaker - likely a voicemail or incomplete call
            role_assignments[unique_speakers[0]] = "unknown"
        
        # Use context hints if available
        if context:
            caller_info = context.get("caller_info", {})
            if caller_info.get("type") == "insurance":
                # Swap roles if insurance called provider
                for speaker_id in role_assignments:
                    if role_assignments[speaker_id] == "provider":
                        role_assignments[speaker_id] = "insurance"
                    elif role_assignments[speaker_id] == "insurance":
                        role_assignments[speaker_id] = "provider"
        
        logger.info(f"Speaker role assignments: {role_assignments}")
        return role_assignments
    
    async def merge_with_transcript(
        self,
        diarization_result: DiarizationResult,
        transcript_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge diarization results with transcript segments.
        
        Args:
            diarization_result: Speaker diarization results
            transcript_segments: Transcript segments with timestamps
            
        Returns:
            Enhanced transcript segments with speaker information
        """
        logger.info("Merging diarization with transcript segments")
        
        enhanced_segments = []
        
        for transcript_segment in transcript_segments:
            transcript_start = transcript_segment.get('start', 0)
            transcript_end = transcript_segment.get('end', 0)
            
            # Find overlapping speaker segment
            best_match = None
            best_overlap = 0
            
            for speaker_segment in diarization_result.segments:
                # Calculate overlap
                overlap_start = max(transcript_start, speaker_segment.start)
                overlap_end = min(transcript_end, speaker_segment.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_match = speaker_segment
            
            # Add speaker information to transcript segment
            enhanced_segment = transcript_segment.copy()
            if best_match and best_overlap > 0:
                enhanced_segment['speaker'] = best_match.speaker_id
                enhanced_segment['speaker_role'] = best_match.speaker_role
                enhanced_segment['speaker_confidence'] = best_match.confidence
            else:
                enhanced_segment['speaker'] = "UNKNOWN"
                enhanced_segment['speaker_role'] = "unknown"
                enhanced_segment['speaker_confidence'] = 0.0
            
            enhanced_segments.append(enhanced_segment)
        
        logger.info(f"Merged {len(enhanced_segments)} transcript segments with speaker information")
        return enhanced_segments
    
    async def analyze_conversation_flow(
        self,
        diarization_result: DiarizationResult
    ) -> Dict[str, Any]:
        """
        Analyze conversation flow patterns for healthcare calls.
        
        Args:
            diarization_result: Diarization results
            
        Returns:
            Conversation flow analysis
        """
        logger.info("Analyzing conversation flow patterns")
        
        # Calculate speaking statistics
        speaker_stats = {}
        total_duration = 0
        
        for segment in diarization_result.segments:
            duration = segment.end - segment.start
            total_duration = max(total_duration, segment.end)
            
            if segment.speaker_id not in speaker_stats:
                speaker_stats[segment.speaker_id] = {
                    "total_time": 0,
                    "turn_count": 0,
                    "avg_turn_length": 0,
                    "role": segment.speaker_role
                }
            
            speaker_stats[segment.speaker_id]["total_time"] += duration
            speaker_stats[segment.speaker_id]["turn_count"] += 1
        
        # Calculate averages
        for speaker_id in speaker_stats:
            stats = speaker_stats[speaker_id]
            stats["avg_turn_length"] = stats["total_time"] / stats["turn_count"] if stats["turn_count"] > 0 else 0
            stats["talk_percentage"] = (stats["total_time"] / total_duration) * 100 if total_duration > 0 else 0
        
        # Analyze turn-taking patterns
        turn_switches = 0
        current_speaker = None
        
        sorted_segments = sorted(diarization_result.segments, key=lambda x: x.start)
        for segment in sorted_segments:
            if current_speaker and current_speaker != segment.speaker_id:
                turn_switches += 1
            current_speaker = segment.speaker_id
        
        analysis = {
            "total_duration": total_duration,
            "num_speakers": diarization_result.num_speakers,
            "turn_switches": turn_switches,
            "speaker_stats": speaker_stats,
            "conversation_balance": self._assess_conversation_balance(speaker_stats),
            "interaction_quality": self._assess_interaction_quality(turn_switches, total_duration)
        }
        
        logger.info(f"Conversation flow analysis completed: {analysis}")
        return analysis
    
    def _assess_conversation_balance(self, speaker_stats: Dict[str, Any]) -> str:
        """Assess if conversation is balanced between speakers."""
        if len(speaker_stats) < 2:
            return "single_speaker"
        
        talk_percentages = [stats["talk_percentage"] for stats in speaker_stats.values()]
        max_percentage = max(talk_percentages)
        
        if max_percentage > 80:
            return "dominated"
        elif max_percentage > 65:
            return "unbalanced"
        else:
            return "balanced"
    
    def _assess_interaction_quality(self, turn_switches: int, total_duration: float) -> str:
        """Assess quality of speaker interaction."""
        if total_duration == 0:
            return "unknown"
        
        turns_per_minute = (turn_switches / total_duration) * 60
        
        if turns_per_minute > 15:
            return "high_interaction"
        elif turns_per_minute > 8:
            return "good_interaction"
        elif turns_per_minute > 3:
            return "moderate_interaction"
        else:
            return "low_interaction"
    
    def export_diarization_json(self, result: DiarizationResult) -> str:
        """
        Export diarization result as JSON.
        
        Args:
            result: Diarization result
            
        Returns:
            JSON string representation
        """
        export_data = {
            "num_speakers": result.num_speakers,
            "processing_time": result.processing_time,
            "speaker_mapping": result.speaker_mapping,
            "segments": [
                {
                    "speaker_id": segment.speaker_id,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.confidence,
                    "speaker_role": segment.speaker_role
                }
                for segment in result.segments
            ]
        }
        
        return json.dumps(export_data, indent=2)
    
    def supports_realtime(self) -> bool:
        """Check if current provider supports real-time diarization."""
        return self.provider.supports_realtime()