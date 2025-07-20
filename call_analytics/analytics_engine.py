import logging
from typing import Dict, List, Any, Optional
import spacy
from spacy.matcher import PhraseMatcher
from spacy import Language
import json
import statistics
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from shared.config import config

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Integrated analytics engine combining objection handling, conversation analysis,
    success prediction, and timing analysis. Uses spaCy for NLP tasks and XGBoost for prediction.
    Designed to be simple and self-contained for processing conversation transcripts and segments.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", xgb_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the engine with spaCy and XGBoost models.

        :param spacy_model: spaCy model name to load.
        :param xgb_params: Optional parameters for XGBoost classifier.
        """
        try:
            self.nlp: Language = spacy.load(spacy_model)
            self.matcher = PhraseMatcher(self.nlp.vocab)
            self._load_objection_patterns()

            default_xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "use_label_encoder": False
            }
            if xgb_params:
                default_xgb_params.update(xgb_params)
            self.xgb_model = xgb.XGBClassifier(**default_xgb_params)

            # Predefined response map (can be updated)
            self.response_map: Dict[str, str] = {
                "cost_objection": "We can discuss payment plans or assistance programs available for this treatment.",
                "eligibility_objection": "Let me provide more details on the patient's condition to confirm eligibility.",
                "documentation_objection": "I can upload the required documents right away; what specifically is missing?",
                "alternative_objection": "The specialist recommends this due to the patient's unique response to alternatives."
            }

            logger.info("AnalyticsEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AnalyticsEngine: {e}")
            raise

    def _load_objection_patterns(self):
        """
        Load predefined objection patterns into the spaCy matcher.
        """
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

    def analyze_conversation(self, transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive conversation analysis including flow, entities, intents,
        objections, timing metrics, and patterns.

        :param transcript: Full transcript text.
        :param segments: List of segment dicts with 'text', 'speaker', 'start', 'end'.
        :return: Dict containing all analysis results.
        """
        try:
            doc = self.nlp(transcript)

            # Extract entities
            entities = {ent.label_: ent.text for ent in doc.ents}

            # Build flow with intents and entities
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

            # Detect objections
            objections = self.detect_objections(transcript)

            # Map responses
            responses = self.map_responses(objections)

            # Analyze timing
            timing_metrics = self.analyze_timing(segments)
            timing_patterns = self.identify_timing_patterns(timing_metrics)

            analysis = {
                "entities": entities,
                "intents": list(set(turn["intent"] for turn in flow if turn["intent"])),
                "flow": flow,
                "objections": objections,
                "responses": responses,
                "timing_metrics": timing_metrics,
                "timing_patterns": timing_patterns
            }

            logger.info("Conversation analysis completed")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            raise

    def _classify_intent(self, text: str) -> Optional[str]:
        """
        Classify intent of a text utterance (rule-based).

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

        :param objections: List of detected objections.
        :return: Dict of objection_text -> response.
        """
        mappings = {}
        for obj in objections:
            obj_type = obj["type"]
            response = self.response_map.get(obj_type, "Generic response: Let's address this concern.")
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

    def analyze_timing(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze timing from transcript segments.

        :param segments: List of segments with 'start', 'end', 'speaker'.
        :return: Dict of timing metrics.
        """
        if not segments:
            raise ValueError("No segments provided")

        segments.sort(key=lambda s: s['start'])

        total_duration = segments[-1]['end'] - segments[0]['start']
        speaker_durations = {}
        response_times = []
        pauses = []

        prev_end = segments[0]['start']
        prev_speaker = None

        for seg in segments:
            # Pause before this segment
            pause = seg['start'] - prev_end
            if pause > 0:
                pauses.append(pause)

            # Segment duration
            dur = seg['end'] - seg['start']
            speaker = seg.get('speaker', 'UNKNOWN')
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + dur

            # Response time if speaker change
            if prev_speaker and speaker != prev_speaker:
                response_time = seg['start'] - prev_end
                response_times.append(response_time)

            prev_end = seg['end']
            prev_speaker = speaker

        metrics = {
            "total_duration": total_duration,
            "speaker_durations": speaker_durations,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "average_pause": statistics.mean(pauses) if pauses else 0,
            "pause_count": len(pauses),
            "response_time_count": len(response_times)
        }

        logger.info("Timing analysis completed")
        return metrics

    def identify_timing_patterns(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Identify timing patterns or issues.

        :param metrics: Timing metrics dict.
        :return: List of pattern descriptions.
        """
        patterns = []
        if metrics["average_response_time"] > 5:  # Arbitrary threshold
            patterns.append("High average response time; potential hesitation.")
        if metrics["pause_count"] > 10:
            patterns.append("Frequent pauses; may indicate uncertainty.")
        if "REP" in metrics["speaker_durations"] and "INS" in metrics["speaker_durations"]:
            rep_ratio = metrics["speaker_durations"]["REP"] / metrics["total_duration"]
            if rep_ratio > 0.6:
                patterns.append("Rep speaking majority of time; good control.")

        logger.info(f"Identified {len(patterns)} timing patterns")
        return patterns

    def extract_features(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from conversation analysis for prediction.

        :param analysis: Dict from analyze_conversation.
        :return: Feature dict.
        """
        flow = analysis.get("flow", [])
        if not flow:
            raise ValueError("No flow data in analysis")

        objection_count = len(analysis.get("objections", []))
        question_count = sum(1 for turn in flow if turn["intent"] == "question")
        agreement_count = sum(1 for turn in flow if turn["intent"] == "agreement")
        total_turns = len(flow)
        duration = analysis["timing_metrics"]["total_duration"]

        features = {
            "objection_count": objection_count,
            "question_count": question_count,
            "agreement_count": agreement_count,
            "turn_count": total_turns,
            "duration": duration,
            "objection_ratio": objection_count / total_turns if total_turns else 0,
            "agreement_ratio": agreement_count / total_turns if total_turns else 0
        }

        logger.info("Features extracted")
        return features

    def predict_success(self, features: Dict[str, float]) -> float:
        """
        Predict success probability for a conversation.

        :param features: Feature dict.
        :return: Success probability (0-1).
        """
        df = pd.DataFrame([features])
        prob = self.xgb_model.predict_proba(df)[0][1]
        logger.info(f"Predicted success probability: {prob:.2f}")
        return prob

    def train_predictor(self, data: List[Dict[str, Any]], labels: List[int]):
        """
        Train the XGBoost model on historical data.

        :param data: List of feature dicts.
        :param labels: List of binary labels (1: success, 0: failure).
        """
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

        self.xgb_model.fit(X_train, y_train)

        # Evaluate
        preds = self.xgb_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model trained with accuracy: {acc:.2f}")