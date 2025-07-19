"""
Transcription Engine - Unified healthcare transcription processing.
Provides speech-to-text, speaker diarization, audio processing, and validation.
"""

from .stt_engine import (
    STTEngine,
    ISTTProvider,
    OpenAIWhisperProvider,
    DeepgramProvider,
    AssemblyAIProvider,
    TranscriptionResult,
    TranscriptionSegment
)

from .audio_processor import (
    AudioProcessor,
    AudioMetrics
)

from .speaker_diarization import (
    SpeakerDiarizer,
    ISpeakerDiarizationProvider,
    PyAnnoteProvider,
    SpeakerSegment,
    DiarizationResult
)

from .transcript_validator import (
    TranscriptValidator,
    ITranscriptValidator,
    LLMTranscriptValidator,
    ValidationResult,
    ValidationIssue,
    validate_transcript_sync
)

from .pipeline import (
    TranscriptionPipeline,
    PipelineConfig,
    PipelineResult
)

__all__ = [
    # STT Engine
    "STTEngine",
    "ISTTProvider", 
    "OpenAIWhisperProvider",
    "DeepgramProvider",
    "AssemblyAIProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    
    # Audio Processing
    "AudioProcessor",
    "AudioMetrics",
    
    # Speaker Diarization
    "SpeakerDiarizer",
    "ISpeakerDiarizationProvider",
    "PyAnnoteProvider", 
    "SpeakerSegment",
    "DiarizationResult",
    
    # Validation
    "TranscriptValidator",
    "ITranscriptValidator",
    "LLMTranscriptValidator",
    "ValidationResult",
    "ValidationIssue",
    "validate_transcript_sync",
    
    # Pipeline
    "TranscriptionPipeline",
    "PipelineConfig",
    "PipelineResult"
]

__version__ = "1.0.0"