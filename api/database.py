# MongoDB database operations for Alfons Prior Authorization Bot
# Replaces Supabase with MongoDB for all conversation and patient data

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import asyncio

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

class DatabaseConnection:
    """Async MongoDB connection manager"""
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            cls._client = AsyncIOMotorClient(settings.MONGO_URI)
            # Test connection
            await cls._client.admin.command('ping')
            logger.info("Connected to MongoDB")
        return cls._client
    
    @classmethod
    async def get_database(cls) -> AsyncIOMotorDatabase:
        if cls._db is None:
            client = await cls.get_client()
            cls._db = client[settings.MONGO_DB_NAME]
        return cls._db
    
    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None

def validate_authorization_data(extracted_data: dict) -> dict:
    """Validate and clean extracted authorization data"""
    valid_statuses = ['approved', 'denied', 'pending', 'under_review', 'requires_documentation']
    
    # Clean approval status
    approval_status = extracted_data.get("approval_status")
    if approval_status:
        approval_status = approval_status.lower().strip()
        if approval_status not in valid_statuses:
            status_mappings = {
                'approve': 'approved', 'deny': 'denied', 'reject': 'denied',
                'rejected': 'denied', 'wait': 'pending', 'waiting': 'pending',
                'review': 'under_review', 'reviewing': 'under_review'
            }
            approval_status = status_mappings.get(approval_status, approval_status)
    
    # Clean authorization number
    auth_number = extracted_data.get("auth_number")
    if auth_number:
        auth_number = auth_number.upper().strip()
        for prefix in ['AUTH-', 'AUTH_', 'AUTH ']:
            if auth_number.startswith(prefix):
                auth_number = auth_number[len(prefix):]
                break
        elif auth_number.startswith('AUTH'):
            auth_number = auth_number[4:]
    
    # Clean patient ID
    patient_id = extracted_data.get("patient_id")
    if patient_id:
        patient_id = patient_id.strip().upper()
    
    # Clean procedure code
    procedure_code = extracted_data.get("procedure_code")
    if procedure_code:
        procedure_code = procedure_code.strip().upper()
        for prefix in ['CPT-', 'CPT_', 'CPT ']:
            if procedure_code.startswith(prefix):
                procedure_code = procedure_code[len(prefix):]
                break
        elif procedure_code.startswith('CPT'):
            procedure_code = procedure_code[3:]
    
    return {
        "patient_id": patient_id,
        "procedure_code": procedure_code,
        "insurance": extracted_data.get("insurance"),
        "approval_status": approval_status,
        "auth_number": auth_number,
        "escalate": extracted_data.get("escalate", False)
    }

async def log_conversation(
    call_sid: str,
    user_input: str,
    bot_response: str,
    extracted_data: dict,
    escalated: bool = False
) -> str:
    """
    Log conversation data to MongoDB
    
    Returns:
        Document ID as string
    """
    validated_data = validate_authorization_data(extracted_data)
    
    document = {
        "call_sid": call_sid,
        "user_input": user_input,
        "bot_response": bot_response,
        "patient_id": validated_data.get("patient_id"),
        "procedure_code": validated_data.get("procedure_code"),
        "insurance": validated_data.get("insurance"),
        "approval_status": validated_data.get("approval_status"),
        "auth_number": validated_data.get("auth_number"),
        "escalated": escalated,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }
    
    try:
        db = await DatabaseConnection.get_database()
        result = await db.conversations.insert_one(document)
        logger.info(f"Successfully logged conversation for call {call_sid}", 
                   extra={"call_sid": call_sid, "doc_id": str(result.inserted_id)})
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to log conversation for call {call_sid}: {str(e)}", 
                    extra={"call_sid": call_sid})
        raise

async def get_logs(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve conversation logs from MongoDB
    
    Args:
        limit: Maximum number of logs to return
        skip: Number of logs to skip (for pagination)
        
    Returns:
        List of conversation log dictionaries
    """
    try:
        db = await DatabaseConnection.get_database()
        cursor = db.conversations.find().sort("timestamp", -1).skip(skip).limit(limit)
        logs = []
        
        async for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime to ISO string
            if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            if "created_at" in doc and isinstance(doc["created_at"], datetime):
                doc["created_at"] = doc["created_at"].isoformat()
                
            logs.append(doc)
        
        logger.info(f"Retrieved {len(logs)} conversation logs")
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve conversation logs: {str(e)}")
        raise

async def get_conversation_by_call_sid(call_sid: str) -> Optional[Dict[str, Any]]:
    """Get all conversations for a specific call SID"""
    try:
        db = await DatabaseConnection.get_database()
        cursor = db.conversations.find({"call_sid": call_sid}).sort("timestamp", 1)
        conversations = []
        
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            conversations.append(doc)
        
        return conversations
    except Exception as e:
        logger.error(f"Failed to get conversations for call {call_sid}: {str(e)}")
        raise

async def update_conversation(doc_id: str, updates: Dict[str, Any]) -> int:
    """Update a conversation document"""
    try:
        db = await DatabaseConnection.get_database()
        updates["updated_at"] = datetime.utcnow()
        
        result = await db.conversations.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": updates}
        )
        
        logger.info(f"Updated conversation {doc_id}, modified: {result.modified_count}")
        return result.modified_count
    except Exception as e:
        logger.error(f"Failed to update conversation {doc_id}: {str(e)}")
        raise

async def delete_conversation(doc_id: str) -> int:
    """Delete a conversation document"""
    try:
        db = await DatabaseConnection.get_database()
        result = await db.conversations.delete_one({"_id": ObjectId(doc_id)})
        
        logger.info(f"Deleted conversation {doc_id}, count: {result.deleted_count}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Failed to delete conversation {doc_id}: {str(e)}")
        raise

# Analytics queries
async def get_conversation_stats(days: int = 30) -> Dict[str, Any]:
    """Get conversation statistics for the last N days"""
    try:
        db = await DatabaseConnection.get_database()
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_date}}},
            {"$group": {
                "_id": None,
                "total_conversations": {"$sum": 1},
                "escalated_count": {"$sum": {"$cond": ["$escalated", 1, 0]}},
                "approved_count": {"$sum": {"$cond": [{"$eq": ["$approval_status", "approved"]}, 1, 0]}},
                "denied_count": {"$sum": {"$cond": [{"$eq": ["$approval_status", "denied"]}, 1, 0]}},
                "pending_count": {"$sum": {"$cond": [{"$eq": ["$approval_status", "pending"]}, 1, 0]}}
            }}
        ]
        
        result = await db.conversations.aggregate(pipeline).to_list(1)
        return result[0] if result else {}
    except Exception as e:
        logger.error(f"Failed to get conversation stats: {str(e)}")
        raise