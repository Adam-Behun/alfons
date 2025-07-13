import logging

# Import core modules from subpackages
from .input_sources.historical_uploader import HistoricalUploader
from .transcription.stt_engine import STTEngine
from .transcription.audio_processor import AudioProcessor
from .transcription.speaker_diarization import SpeakerDiarizer
from .transcription.transcript_validator import TranscriptValidator
from .analytics.conversation_analyzer import ConversationAnalyzer
from .analytics.success_predictor import SuccessPredictor
from .analytics.objection_handler import ObjectionHandler
from .analytics.timing_analyzer import TimingAnalyzer
from .learning.pattern_extractor import PatternExtractor
from .learning.script_generator import ScriptGenerator
from .learning.training_pipeline import TrainingPipeline
from .learning.memory_manager import MemoryManager
from .database.mongo_connector import MongoConnector
from .database.sql_models import Base, Recording, Analytic, Pattern  # If SQL is used
from .database.vector_index import VectorIndex
from .web_interface.upload_dashboard import run_upload_dashboard
from .web_interface.analytics_dashboard import run_analytics_dashboard
from .web_interface.training_dashboard import run_training_dashboard
from .api.upload_api import app as upload_app
from .api.analytics_api import app as analytics_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.info("call_analytics package initialized")