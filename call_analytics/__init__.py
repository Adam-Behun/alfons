"""
Simplified call_analytics module exports.
Exports key classes from merged files for MVP use.
"""

from .analytics_engine import AnalyticsEngine
from .database_connector import MongoConnector
from .input_handler import InputHandler
from .learning_pipeline import PatternExtractor, ScriptGenerator, MemoryManager, TrainingPipeline
from .queue_manager import process_call, get_queue_status, health_check  # From simplified task_queue.py
from .transcription_pipeline import TranscriptionPipeline, TranscriptionSegment, TranscriptionResult

__all__ = [
    "AnalyticsEngine",
    "MongoConnector",
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