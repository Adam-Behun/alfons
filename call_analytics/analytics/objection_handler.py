import logging
from typing import Dict, List, Any, Optional
import spacy
from spacy.matcher import PhraseMatcher
import json

from config.settings import settings
from ..learning.pattern_extractor import PatternExtractor  # For integration if needed

logger = logging.getLogger(__name__)

class ObjectionHandler:
    """
    Handles objection pattern matching and response mapping.
    Identifies common objections in conversations and suggests handling responses.
    Uses spaCy for pattern matching; can integrate with LLM for dynamic responses.
    Maintains a dictionary of known objections and best responses from historical data.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the handler with a spaCy model and matcher.
        
        :param model_name: spaCy model to load.
        """
        try:
            self.nlp = spacy.load(model_name)
            self.matcher = PhraseMatcher(self.nlp.vocab)
            self._load_patterns()
            self.response_map: Dict[str, str] = {}  # objection_key -> response
            logger.info("ObjectionHandler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ObjectionHandler: {e}")
            raise
    
    def _load_patterns(self):
        """
        Load predefined objection patterns into the matcher.
        Patterns can be extended from memory or training data.
        """
        # Example patterns; load from DB in production
        patterns = {
            "cost_objection": ["too expensive", "cost too much", "budget issue"],
            "eligibility_objection": ["not eligible", "doesn't qualify", "criteria not met"],
            "documentation_objection": ["need more documents", "missing paperwork", "insufficient info"],
            "alternative_objection": ["try alternative", "generic drug", "other treatment"]
        }
        
        for key, phrases in patterns.items():
            phrase_patterns = [self.nlp(phrase) for phrase in phrases]
            self.matcher.add(key, phrase_patterns)
        
        logger.info("Objection patterns loaded")
    
    def detect_objections(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Detect objections in the transcript.
        
        :param transcript: Full transcript text.
        :return: List of detected objections with type, text, position.
        """
        doc = self.nlp(transcript)
        matches = self.matcher(doc)
        
        objections = []
        for match_id, start, end in matches:
            objection_type = self.nlp.vocab.strings[match_id]
            text = doc[start:end].text
            objections.append({
                "type": objection_type,
                "text": text,
                "start": doc[start].idx,
                "end": doc[end-1].idx + len(doc[end-1].text)
            })
        
        logger.info(f"Detected {len(objections)} objections")
        return objections
    
    def map_responses(self, objections: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Map detected objections to recommended responses.
        For MVP, use predefined responses; integrate with script_generator.py later.
        
        :param objections: List of detected objections.
        :return: Dict of objection_text -> response.
        """
        # Predefined responses; load from memory/DB
        predefined_responses = {
            "cost_objection": "We can discuss payment plans or assistance programs available for this treatment.",
            "eligibility_objection": "Let me provide more details on the patient's condition to confirm eligibility.",
            "documentation_objection": "I can upload the required documents right away; what specifically is missing?",
            "alternative_objection": "The specialist recommends this due to the patient's unique response to alternatives."
        }
        
        mappings = {}
        for obj in objections:
            obj_type = obj["type"]
            response = predefined_responses.get(obj_type, "Generic response: Let's address this concern.")
            mappings[obj["text"]] = response
        
        logger.info(f"Mapped responses for {len(mappings)} objections")
        return mappings
    
    def update_response_map(self, new_map: Dict[str, str]):
        """
        Update the response map with new learnings.
        
        :param new_map: Dict of objection_key -> response.
        """
        self.response_map.update(new_map)
        logger.info("Response map updated")

# Example usage (for testing)
if __name__ == "__main__":
    handler = ObjectionHandler()
    try:
        mock_transcript = "The cost is too expensive and the patient doesn't qualify."
        objections = handler.detect_objections(mock_transcript)
        print(json.dumps(objections, indent=2))
        
        responses = handler.map_responses(objections)
        print(json.dumps(responses, indent=2))
    except Exception as e:
        print(f"Error: {e}")