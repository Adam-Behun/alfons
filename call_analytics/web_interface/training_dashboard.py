import streamlit as st
import logging
import pandas as pd
from typing import List, Dict, Any
import json

from call_analytics.config.settings import settings
from ..database.mongo_connector import MongoConnector
from ..learning.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

def run_training_dashboard():
    """
    Streamlit dashboard for reviewing and exporting training data.
    Displays annotated data, patterns, scripts; allows export to JSON/CSV.
    Integrates with TrainingPipeline for data prep.
    """
    st.title("Alfons Training Data Review Dashboard")
    
    connector = MongoConnector()
    pipeline = TrainingPipeline()
    
    try:
        # Fetch annotated data (assume stored in 'training_data' collection; or run pipeline)
        st.subheader("Load Data")
        if st.button("Refresh Data from Pipeline"):
            pipeline.run_pipeline()  # Runs collect, annotate, format, but not export
            st.success("Data refreshed!")
        
        training_data: List[Dict[str, Any]] = connector.find_documents("training_data", limit=0)  # Assume stored
        if not training_data:
            st.warning("No training data available. Run pipeline first.")
            return
        
        df = pd.DataFrame(training_data)
        
        # Display data table
        st.subheader("Training Data Preview")
        st.dataframe(df.head(10))  # Show first 10
        
        # Review specific item
        st.subheader("Review Item")
        item_id = st.selectbox("Select Item ID", df["_id"].tolist())
        if item_id:
            item = connector.find_documents("training_data", {"_id": item_id}, limit=1)[0]
            st.json(item)
            
            # Display patterns
            patterns = item.get("patterns", [])
            if patterns:
                st.subheader("Extracted Patterns")
                patterns_df = pd.DataFrame(patterns)
                st.dataframe(patterns_df)
            
            # Display generated script
            script = item.get("generated_script")
            if script:
                st.subheader("Generated Script")
                st.text_area("Script", script, height=200)
        
        # Export options
        st.subheader("Export Data")
        format_type = st.selectbox("Format", ["json", "csv"])
        if st.button("Export"):
            export_path = f"exported_data.{format_type}"
            pipeline.split_and_export(df, export_path)
            st.success(f"Data exported to {export_path}_train.{format_type} and {export_path}_test.{format_type}")
            # Provide download links
            with open(f"{export_path}_train.{format_type}", "rb") as f:
                st.download_button("Download Train", f, file_name=f"train.{format_type}")
            with open(f"{export_path}_test.{format_type}", "rb") as f:
                st.download_button("Download Test", f, file_name=f"test.{format_type}")
    
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")
    finally:
        connector.close_connection()

if __name__ == "__main__":
    run_training_dashboard()