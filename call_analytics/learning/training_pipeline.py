import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from call_analytics.config.settings import settings
from ..database.mongo_connector import MongoConnector
from .pattern_extractor import PatternExtractor
from .script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Prepares training data for fine-tuning or RAG.
    Annotates and formats data from transcripts, analytics, patterns.
    Generates datasets (e.g., for LLM fine-tuning: prompt-response pairs).
    Splits data into train/val/test; exports to JSON/CSV.
    Integrates with extractor and generator for automated labeling.
    """
    
    def __init__(self):
        """
        Initialize the training pipeline with MongoDB connector.
        """
        self.connector = MongoConnector()
        self.extractor = PatternExtractor()  # Assumes initialized with key
        self.generator = ScriptGenerator()  # Assumes initialized with key
        logger.info("TrainingPipeline initialized")
    
    def collect_data(self, collection_name: str = "transcripts", query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Collect data from MongoDB.
        
        :param collection_name: Collection to query (e.g., 'transcripts', 'analytics').
        :param query: Optional query filter.
        :return: List of documents.
        """
        query = query or {}
        data = self.connector.find_documents(collection_name, query)
        logger.info(f"Collected {len(data)} documents from {collection_name}")
        return data
    
    def annotate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate data using LLM (extract patterns, generate labels/responses).
        
        :param data: List of raw documents (e.g., with 'transcript').
        :return: Annotated data with patterns, scripts, labels.
        """
        annotated = []
        for item in data:
            transcript = item.get("transcript", "")
            if not transcript:
                continue
            
            patterns = self.extractor.extract_patterns(transcript)
            item["patterns"] = patterns
            
            # Generate sample script for training (e.g., response to context)
            context = " ".join([p["phrase"] for p in patterns if p["type"] == "objection_handling"])
            if context:
                script = self.generator.generate_script(context)
                item["generated_script"] = script
            
            # Add label (assume 'outcome' field; else, predict or manual)
            item["label"] = 1 if item.get("outcome") == "success" else 0
            
            annotated.append(item)
        
        logger.info(f"Annotated {len(annotated)} items")
        return annotated
    
    def format_data(self, annotated: List[Dict[str, Any]], format_type: str = "fine_tune") -> pd.DataFrame:
        """
        Format annotated data into a DataFrame for training.
        
        :param annotated: Annotated data list.
        :param format_type: 'fine_tune' (prompt-response) or 'rag' (context-embedding).
        :return: Formatted DataFrame.
        """
        if format_type == "fine_tune":
            rows = []
            for item in annotated:
                prompt = f"Transcript: {item['transcript']}\nPatterns: {json.dumps(item['patterns'])}"
                response = item.get("generated_script", "")
                label = item["label"]
                rows.append({"prompt": prompt, "response": response, "label": label})
            df = pd.DataFrame(rows)
        elif format_type == "rag":
            # For RAG: context, question, answer
            df = pd.DataFrame([{"context": item["transcript"], "patterns": item["patterns"]} for item in annotated])
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        
        logger.info(f"Formatted data into DataFrame with {len(df)} rows")
        return df
    
    def split_and_export(self, df: pd.DataFrame, export_path: str = "training_data.json", test_size: float = 0.2):
        """
        Split data and export to file.
        
        :param df: Formatted DataFrame.
        :param export_path: Path to export (JSON or CSV based on extension).
        :param test_size: Test split ratio.
        """
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        if export_path.endswith(".json"):
            train_df.to_json(export_path.replace(".json", "_train.json"), orient="records", indent=4)
            test_df.to_json(export_path.replace(".json", "_test.json"), orient="records", indent=4)
        elif export_path.endswith(".csv"):
            train_df.to_csv(export_path.replace(".csv", "_train.csv"), index=False)
            test_df.to_csv(export_path.replace(".csv", "_test.csv"), index=False)
        else:
            raise ValueError("Unsupported export format")
        
        logger.info(f"Exported training data to {export_path}")
    
    def run_pipeline(self, collection_name: str = "transcripts"):
        """
        Run the full pipeline: collect, annotate, format, split/export.
        
        :param collection_name: Source collection.
        """
        data = self.collect_data(collection_name)
        annotated = self.annotate_data(data)
        df = self.format_data(annotated)
        self.split_and_export(df)

# Example usage (for testing)
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    try:
        # Assuming some data in DB; mock if needed
        pipeline.run_pipeline()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.connector.close_connection()