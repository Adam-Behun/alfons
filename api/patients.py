# Patient database functions for Alfons Prior Authorization Bot
# MongoDB implementation for patient data operations

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from bson import ObjectId

from shared.config import settings
from shared.logging import get_logger
from .database import DatabaseConnection

logger = get_logger(__name__)

async def create_patient(patient_data: Dict[str, Any]) -> str:
    """Create a new patient record"""
    try:
        # Add metadata
        patient_data.update({
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "prior_auth_status": patient_data.get("prior_auth_status", "Pending")
        })
        
        db = await DatabaseConnection.get_database()
        result = await db.patients.insert_one(patient_data)
        
        logger.info(f"Created patient with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to create patient: {str(e)}")
        raise

async def get_patients(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Retrieve all patients from the database"""
    try:
        db = await DatabaseConnection.get_database()
        cursor = db.patients.find().sort("appointment_time", 1).skip(skip).limit(limit)
        patients = []
        
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime fields to ISO strings
            for field in ["appointment_time", "date_of_birth", "created_at", "updated_at"]:
                if field in doc and isinstance(doc[field], datetime):
                    doc[field] = doc[field].isoformat()
            
            patients.append(doc)
        
        logger.info(f"Retrieved {len(patients)} patients")
        return patients
    except Exception as e:
        logger.error(f"Failed to retrieve patients: {str(e)}")
        raise

async def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific patient by ID"""
    try:
        db = await DatabaseConnection.get_database()
        doc = await db.patients.find_one({"_id": ObjectId(patient_id)})
        
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime fields
            for field in ["appointment_time", "date_of_birth", "created_at", "updated_at"]:
                if field in doc and isinstance(doc[field], datetime):
                    doc[field] = doc[field].isoformat()
        
        return doc
    except Exception as e:
        logger.error(f"Failed to get patient {patient_id}: {str(e)}")
        raise

async def get_patient_by_member_id(member_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve patient by insurance member ID"""
    try:
        db = await DatabaseConnection.get_database()
        doc = await db.patients.find_one({"insurance_member_id": member_id})
        
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime fields
            for field in ["appointment_time", "date_of_birth", "created_at", "updated_at"]:
                if field in doc and isinstance(doc[field], datetime):
                    doc[field] = doc[field].isoformat()
        
        return doc
    except Exception as e:
        logger.error(f"Failed to get patient by member ID {member_id}: {str(e)}")
        raise

async def update_patient_auth_status(patient_id: str, status: str, auth_number: str = None) -> int:
    """Update the prior authorization status for a patient"""
    try:
        updates = {
            "prior_auth_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if auth_number:
            updates["auth_number"] = auth_number
        
        db = await DatabaseConnection.get_database()
        result = await db.patients.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": updates}
        )
        
        logger.info(f"Updated patient {patient_id} auth status to {status}")
        return result.modified_count
    except Exception as e:
        logger.error(f"Failed to update patient auth status: {str(e)}")
        raise

async def get_pending_authorizations() -> List[Dict[str, Any]]:
    """Get all patients with pending prior authorization status"""
    try:
        db = await DatabaseConnection.get_database()
        cursor = db.patients.find({"prior_auth_status": "Pending"}).sort("appointment_time", 1)
        patients = []
        
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime fields
            for field in ["appointment_time", "date_of_birth", "created_at", "updated_at"]:
                if field in doc and isinstance(doc[field], datetime):
                    doc[field] = doc[field].isoformat()
            
            patients.append(doc)
        
        logger.info(f"Retrieved {len(patients)} pending authorizations")
        return patients
    except Exception as e:
        logger.error(f"Failed to get pending authorizations: {str(e)}")
        raise

async def get_patients_by_provider(provider_npi: str) -> List[Dict[str, Any]]:
    """Get all patients for a specific provider"""
    try:
        db = await DatabaseConnection.get_database()
        cursor = db.patients.find({"provider_npi": provider_npi}).sort("appointment_time", 1)
        patients = []
        
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Convert datetime fields
            for field in ["appointment_time", "date_of_birth", "created_at", "updated_at"]:
                if field in doc and isinstance(doc[field], datetime):
                    doc[field] = doc[field].isoformat()
            
            patients.append(doc)
        
        return patients
    except Exception as e:
        logger.error(f"Failed to get patients for provider {provider_npi}: {str(e)}")
        raise

async def delete_patient(patient_id: str) -> int:
    """Delete a patient record"""
    try:
        db = await DatabaseConnection.get_database()
        result = await db.patients.delete_one({"_id": ObjectId(patient_id)})
        
        logger.info(f"Deleted patient {patient_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Failed to delete patient {patient_id}: {str(e)}")
        raise