import streamlit as st
import logging
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any

from call_analytics.config.settings import settings
from ..database.mongo_connector import MongoConnector
from ..analytics.success_predictor import SuccessPredictor
from ..analytics.timing_analyzer import TimingAnalyzer
# Assume other analyzers as needed

logger = logging.getLogger(__name__)

def run_analytics_dashboard():
    """
    Streamlit dashboard for analytics visualizations using Plotly.
    Displays charts for success predictions, timing metrics, objection patterns, etc.
    Retrieves data from MongoDB.
    """
    st.title("Alfons Analytics Dashboard")
    
    connector = MongoConnector()
    predictor = SuccessPredictor()
    timing_analyzer = TimingAnalyzer()
    
    try:
        # Fetch data
        analytics_data: List[Dict[str, Any]] = connector.find_documents("analytics", limit=0)  # All for demo
        if not analytics_data:
            st.warning("No analytics data available.")
            return
        
        df = pd.DataFrame(analytics_data)
        
        # Success Prediction Chart
        st.subheader("Success Predictions")
        # Mock features; in practice, extract from data
        features_list = []  # Populate with actual features
        for item in analytics_data:
            # Assume item has 'analysis' from conversation_analyzer
            analysis = item.get("analysis", {})
            features = predictor.extract_features(analysis)
            features_list.append(features)
        
        preds = [predictor.predict(f) for f in features_list]
        df["success_prob"] = preds
        fig_success = px.bar(df, x="_id", y="success_prob", title="Success Probability per Call")
        st.plotly_chart(fig_success)
        
        # Timing Metrics Chart
        st.subheader("Timing Metrics")
        timing_metrics = []
        for item in analytics_data:
            segments = item.get("segments", [])  # Assume stored
            if segments:
                metrics = timing_analyzer.analyze_timing(segments)
                timing_metrics.append(metrics)
        
        if timing_metrics:
            timing_df = pd.DataFrame(timing_metrics)
            fig_timing = px.line(timing_df, x=timing_df.index, y="average_response_time", title="Average Response Time")
            st.plotly_chart(fig_timing)
        
        # Other charts (e.g., objections)
        st.subheader("Objection Counts")
        # Similar logic; mock for now
        obj_fig = px.pie(values=[10, 20, 15], names=["Cost", "Eligibility", "Documentation"], title="Objection Types")
        st.plotly_chart(obj_fig)
    
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")
    finally:
        connector.close_connection()

if __name__ == "__main__":
    run_analytics_dashboard()