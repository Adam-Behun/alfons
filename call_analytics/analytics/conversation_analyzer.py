import logging
from typing import Dict, List, Any, Optional
import spacy
from spacy import Language
import json

from config.settings import settings
from ..transcription.stt_engine import STTEngine  # For reference if needed

logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """
    Analyzes conversation flow using NLP (spaCy/LLM).
    Extracts intents, entities, and maps conversation structure.
    Identifies key elements like objections, agreements, questions.
    Outputs structured data for further analytics.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the analyzer with a spaCy model.
        
        :param model_name: spaCy model to load.
        """
        try:
            self.nlp: Language = spacy.load(model_name)
            logger.info(f"ConversationAnalyzer initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
    
    def analyze_flow(self, transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the conversation flow.
        
        :param transcript: Full transcript text.
        :param segments: List of segment dicts with 'text', 'speaker', 'start', 'end'.
        :return: Analysis dict with intents, entities, flow map.
        """
        try:
            doc = self.nlp(transcript)
            
            # Extract entities
            entities = {ent.label_: ent.text for ent in doc.ents}
            
            # Extract intents (simple rule-based for MVP; extend with LLM)
            intents = self._extract_intents(segments)
            
            # Build flow map: list of turns with speaker, intent, entities
            flow = []
            for seg in segments:
                seg_doc = self.nlp(seg['text'])
                seg_entities = [(ent.text, ent.label_) for ent in seg_doc.ents]
                flow.append({
                    "speaker": seg.get('speaker', 'UNKNOWN'),
                    "text": seg['text'],
                    "intent": self._classify_intent(seg['text']),
                    "entities": seg_entities,
                    "start": seg['start'],
                    "end": seg['end']
                })
            
            analysis = {
                "entities": entities,
                "intents": intents,
                "flow": flow
            }
            
            logger.info("Conversation analysis completed")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            raise
    
    def _extract_intents(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Extract unique intents from segments.
        
        :param segments: Transcript segments.
        :return: List of unique intents.
        """
        intents = set()
        for seg in segments:
            intent = self._classify_intent(seg['text'])
            if intent:
                intents.add(intent)
        return list(intents)
    
    def _classify_intent(self, text: str) -> Optional[str]:
        """
        Classify intent of a text utterance (rule-based for MVP).
        
        :param text: Utterance text.
        :return: Intent string or None.
        """
        text_lower = text.lower()
        if '?' in text:
            return "question"
        elif any(word in text_lower for word in ["yes", "agree", "approve"]):
            return "agreement"
        elif any(word in text_lower for word in ["no", "deny", "object"]):
            return "objection"
        elif "prior authorization" in text_lower:
            return "pa_request"
        else:
            return "statement"

# Example usage (for testing)
if __name__ == "__main__":
    analyzer = ConversationAnalyzer()
    try:
        # Mock data
        mock_transcript = "Rep: We need prior authorization for the drug. Insurance: What is the patient's condition?"
        mock_segments = [
            {"text": "We need prior authorization for the drug.", "speaker": "REP", "start": 0, "end": 5},
            {"text": "What is the patient's condition?", "speaker": "INS", "start": 5, "end": 10}
        ]
        result = analyzer.analyze_flow(mock_transcript, mock_segments)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")