import logging
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from config.settings import settings
from ..database.mongo_connector import MongoConnector
from ..analytics.success_predictor import SuccessPredictor
# Import other analyzers as needed, e.g., TimingAnalyzer

logger = logging.getLogger(__name__)

app = FastAPI(title="Alfons Analytics API")

connector = MongoConnector()
predictor = SuccessPredictor()
# timing_analyzer = TimingAnalyzer()  # If needed

class AnalyticsResponse(BaseModel):
    id: str
    success_prob: Optional[float]
    # Add other fields like timing_metrics: Dict[str, Any]

@app.get("/analytics", response_model=List[Dict[str, Any]])
async def get_all_analytics(limit: int = 10):
    """
    Get a list of all analytics data (limited).
    
    :param limit: Max number of results.
    :return: List of analytics documents.
    """
    try:
        analytics = connector.find_documents("analytics", limit=limit)
        return analytics
    except Exception as e:
        logger.error(f"Get all analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{item_id}", response_model=Dict[str, Any])
async def get_analytics(item_id: str):
    """
    Get analytics for a specific item by ID.
    
    :param item_id: Analytics document ID.
    :return: Analytics dict.
    """
    try:
        analytics = connector.find_documents("analytics", {"_id": item_id}, limit=1)
        if not analytics:
            raise HTTPException(status_code=404, detail="Analytics not found")
        
        item = analytics[0]
        # Enhance with prediction if needed
        analysis = item.get("analysis", {})
        features = predictor.extract_features(analysis)
        item["success_prob"] = predictor.predict(features)
        
        # Add timing if segments available
        # segments = item.get("segments", [])
        # if segments:
        #     item["timing_metrics"] = timing_analyzer.analyze_timing(segments)
        
        return item
    except Exception as e:
        logger.error(f"Get analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)