import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class SuccessPredictor:
    """
    Predicts call success using XGBoost on extracted features.
    Features: objection count, response times, intent ratios, etc.
    Trains on historical data; predicts for new calls.
    For MVP, includes simple training pipeline.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor with XGBoost model.
        
        :param model_params: Optional params for XGBClassifier.
        """
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
        if model_params:
            default_params.update(model_params)
        
        self.model = xgb.XGBClassifier(**default_params)
        logger.info("SuccessPredictor initialized")
    
    def extract_features(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from conversation analysis.
        
        :param analysis: Dict from ConversationAnalyzer.
        :return: Feature dict.
        """
        flow = analysis.get("flow", [])
        if not flow:
            raise ValueError("No flow data in analysis")
        
        objection_count = sum(1 for turn in flow if turn["intent"] == "objection")
        question_count = sum(1 for turn in flow if turn["intent"] == "question")
        agreement_count = sum(1 for turn in flow if turn["intent"] == "agreement")
        total_turns = len(flow)
        duration = flow[-1]["end"] - flow[0]["start"] if flow else 0
        
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
    
    def train(self, data: List[Dict[str, Any]], labels: List[int]):
        """
        Train the model on historical data.
        
        :param data: List of feature dicts.
        :param labels: List of binary labels (1: success, 0: failure).
        """
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model trained with accuracy: {acc:.2f}")
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict success probability for a new call.
        
        :param features: Feature dict.
        :return: Success probability (0-1).
        """
        df = pd.DataFrame([features])
        prob = self.model.predict_proba(df)[0][1]
        logger.info(f"Predicted success probability: {prob:.2f}")
        return prob

# Example usage (for testing)
if __name__ == "__main__":
    predictor = SuccessPredictor()
    try:
        # Mock training data
        mock_data = [
            {"objection_count": 2, "question_count": 3, "agreement_count": 1, "turn_count": 10, "duration": 300, "objection_ratio": 0.2, "agreement_ratio": 0.1},
            {"objection_count": 0, "question_count": 1, "agreement_count": 2, "turn_count": 8, "duration": 200, "objection_ratio": 0.0, "agreement_ratio": 0.25}
        ]
        mock_labels = [0, 1]
        predictor.train(mock_data, mock_labels)
        
        # Mock predict
        mock_features = {"objection_count": 1, "question_count": 2, "agreement_count": 1, "turn_count": 9, "duration": 250, "objection_ratio": 0.11, "agreement_ratio": 0.11}
        prob = predictor.predict(mock_features)
        print(f"Success prob: {prob}")
    except Exception as e:
        print(f"Error: {e}")