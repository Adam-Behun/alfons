"""
Learning hook for capturing and queuing post-call data for async analysis.
Integrates with Twilio webhooks to trigger data processing when calls end.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import HTTPException
from pydantic import BaseModel

# Import task queue for async processing
try:
    from call_analytics.task_queue import process_call
except ImportError:
    # Fallback if task queue not available yet
    process_call = None
    logging.warning("Task queue not available - calls will not be processed asynchronously")

from call_analytics.mongo_connector import AsyncMongoConnector

logger = logging.getLogger(__name__)

class CallEndPayload(BaseModel):
    """Twilio webhook payload structure for call completion"""
    CallSid: str
    CallStatus: str
    CallDuration: Optional[str] = None
    RecordingUrl: Optional[str] = None
    From: Optional[str] = None
    To: Optional[str] = None
    Direction: Optional[str] = None

class LearningData(BaseModel):
    """Structured data for learning pipeline"""
    call_id: str
    timestamp: str
    audio_url: Optional[str] = None
    transcript: Optional[str] = None
    entities: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    escalation: Dict[str, Any] = {}
    raw_payload: Dict[str, Any] = {}
    thoughts: list[str] = []  # Aggregated bot thoughts for learning/auditing

def anonymize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonymize PII in extracted entities for HIPAA compliance.
    Redacts patient IDs and other sensitive information.
    """
    anonymized = entities.copy()
    
    # Redact patient ID if present
    if "patient_id" in anonymized and anonymized["patient_id"]:
        anonymized["patient_id"] = "REDACTED"
    
    # Keep procedure codes and insurance for learning (non-PII)
    # Add more redaction rules as needed
    
    return anonymized

def extract_call_metrics(payload: CallEndPayload) -> Dict[str, Any]:
    """
    Extract call quality and performance metrics from Twilio payload.
    """
    metrics = {}
    
    if payload.CallDuration:
        try:
            metrics["duration_seconds"] = int(payload.CallDuration)
        except (ValueError, TypeError):
            metrics["duration_seconds"] = 0
    
    metrics["direction"] = payload.Direction or "unknown"
    metrics["status"] = payload.CallStatus
    metrics["timestamp"] = datetime.utcnow().isoformat()
    
    return metrics

def prepare_learning_data(
    payload: CallEndPayload,
    conversation_entities: Optional[Dict[str, Any]] = None,
    escalation_data: Optional[Dict[str, Any]] = None,
    thoughts_data: Optional[list[str]] = None
) -> LearningData:
    """
    Prepare structured data for the learning pipeline.
    Combines Twilio payload with conversation analysis results.
    """
    # Generate unique call ID for tracking
    call_id = str(uuid.uuid4())
    
    # Extract and anonymize entities
    entities = conversation_entities or {}
    anonymized_entities = anonymize_entities(entities)
    
    # Extract call metrics
    metrics = extract_call_metrics(payload)
    
    # Prepare escalation data
    escalation_info = escalation_data or {}
    if "escalate" not in escalation_info:
        escalation_info["triggered"] = False
    
    return LearningData(
        call_id=call_id,
        timestamp=datetime.utcnow().isoformat(),
        audio_url=payload.RecordingUrl,
        entities=anonymized_entities,
        metrics=metrics,
        escalation=escalation_info,
        raw_payload=payload.dict(),
        thoughts=thoughts_data or []
    )

async def queue_call_for_processing(learning_data: LearningData) -> bool:
    """
    Queue call data for async processing.
    Returns True if successfully queued, False otherwise.
    """
    if not process_call:
        logger.error("Task queue not available - cannot process call data")
        return False
    
    try:
        # Queue the data for async processing
        task = process_call.delay(learning_data.dict())
        logger.info(f"Queued call {learning_data.call_id} for processing. Task ID: {task.id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to queue call {learning_data.call_id}: {str(e)}")
        return False

async def handle_call_completion(
    payload: CallEndPayload,
    conversation_entities: Optional[Dict[str, Any]] = None,
    escalation_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point for handling call completion.
    Triggered by Twilio webhooks or internal call end events.
    """
    logger.info(f"Processing call completion for CallSid: {payload.CallSid}")
    
    try:
        # Fetch conversation data if not provided
        if conversation_entities is None or escalation_data is None:
            conv_data = await get_conversation_data(payload.CallSid)
            if conv_data:
                conversation_entities = conv_data.get("entities", {})
                escalation_data = conv_data.get("escalation", {})
                thoughts_data = conv_data.get("thoughts", [])
            else:
                conversation_entities = {}
                escalation_data = {}
                thoughts_data = []
        else:
            thoughts_data = []  # If provided externally, assume no thoughts or handle separately

        # Prepare learning data
        learning_data = prepare_learning_data(
            payload, 
            conversation_entities, 
            escalation_data,
            thoughts_data
        )
        
        # Queue for async processing
        success = await queue_call_for_processing(learning_data)
        
        if success:
            return {
                "status": "success",
                "call_id": learning_data.call_id,
                "message": "Call data queued for processing"
            }
        else:
            return {
                "status": "error",
                "call_id": learning_data.call_id,
                "message": "Failed to queue call data"
            }
            
    except Exception as e:
        logger.error(f"Error handling call completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process call completion: {str(e)}")

# Utility function to get conversation data from database
async def get_conversation_data(call_sid: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve conversation entities and escalation data from database.
    This should be called before queuing to get the full conversation context.
    """
    try:
        mongo = AsyncMongoConnector()
        await mongo.connect()
        collection = await mongo.get_collection("logs")
        cursor = collection.find({"call_sid": call_sid}).sort("timestamp", 1)
        logs = await cursor.to_list(length=100)
        await mongo.close_connection()
        
        if logs:
            # Aggregate all conversation data for this call
            entities = {}
            escalation_data = {"triggered": False}
            thoughts = []
            
            for log in logs:
                # Merge entities from all conversation turns
                if log.get("patient_id"):
                    entities["patient_id"] = log["patient_id"]
                if log.get("procedure_code"):
                    entities["procedure_code"] = log["procedure_code"]
                if log.get("insurance"):
                    entities["insurance"] = log["insurance"]
                
                # Check for escalation
                if log.get("escalated"):
                    escalation_data["triggered"] = True
                    escalation_data["reason"] = log.get("bot_response", "")
                
                # Aggregate thoughts
                if log.get("bot_thoughts"):
                    thoughts.append(log["bot_thoughts"])
            
            return {
                "entities": entities,
                "escalation": escalation_data,
                "thoughts": thoughts
            }
            
    except Exception as e:
        logger.error(f"Failed to retrieve conversation data for {call_sid}: {str(e)}")
    
    return None