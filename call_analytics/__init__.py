"""
Exports key classes from merged files for MVP use.
"""

from .analytics_engine import AnalyticsEngine
from .mongo_connector import AsyncMongoConnector
from .input_handler import InputHandler
from .learning_pipeline import PatternExtractor, ScriptGenerator, MemoryManager, TrainingPipeline
from .task_queue import process_call, get_queue_status, health_check
from .transcription_pipeline import TranscriptionPipeline, TranscriptionSegment, TranscriptionResult

__all__ = [
    "AnalyticsEngine",
    "AsyncMongoConnector",
    "InputHandler",
    "PatternExtractor",
    "ScriptGenerator",
    "MemoryManager",
    "TrainingPipeline",
    "process_call",
    "get_queue_status",
    "health_check",
    "TranscriptionPipeline",
    "TranscriptionSegment",
    "TranscriptionResult"
]

__version__ = "1.0.0-mvp"