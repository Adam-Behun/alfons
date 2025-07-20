# Enhanced async MongoDB connector for Alfons analytics system
# Consolidated from database.py, mongo_connector.py, and patients.py for unified connection and operations

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, PyMongoError
from bson.objectid import ObjectId
from datetime import datetime, timedelta

from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)

class AsyncMongoConnector:
    """
    Async MongoDB connector for analytics, bot, and patient operations.
    Handles connection, generic CRUD, aggregations, and domain-specific functions with proper error handling.
    """
    
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self):
        """Initialize async MongoDB connection if not already connected."""
        if self._client is None:
            try:
                self._client = AsyncIOMotorClient(config.MONGO_URI)
                # Test connection
                await self._client.admin.command('ping')
                self._db = self._client[config.MONGO_DB_NAME]
                logger.info(f"Connected to MongoDB database: {config.MONGO_DB_NAME}")
            except ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
    
    @property
    async def client(self) -> AsyncIOMotorClient:
        await self.connect()
        return self._client
    
    @property
    async def db(self) -> AsyncIOMotorDatabase:
        await self.connect()
        return self._db
    
    async def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get async collection reference."""
        db = await self.db
        return db[collection_name]
    
    # Generic CRUD Operations
    
    async def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert single document and return ID as string."""
        try:
            collection = await self.get_collection(collection_name)
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            result = await collection.insert_one(document)
            logger.info(f"Inserted document with ID: {result.inserted_id} in {collection_name}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error inserting document in {collection_name}: {e}")
            raise
    
    async def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents and return IDs as strings."""
        try:
            collection = await self.get_collection(collection_name)
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow()
            result = await collection.insert_many(documents)
            inserted_ids = [str(_id) for _id in result.inserted_ids]
            logger.info(f"Inserted {len(inserted_ids)} documents in {collection_name}")
            return inserted_ids
        except PyMongoError as e:
            logger.error(f"Error inserting multiple documents in {collection_name}: {e}")
            raise
    
    async def find_documents(
        self, 
        collection_name: str, 
        query: Dict[str, Any] = {}, 
        limit: int = 0,
        skip: int = 0,
        sort: Optional[List[tuple]] = None
    ) -> List[Dict[str, Any]]:
        """Find documents with optional pagination and sorting."""
        try:
            collection = await self.get_collection(collection_name)
            cursor = collection.find(query)
            if sort:
                cursor = cursor.sort(sort)
            if skip > 0:
                cursor = cursor.skip(skip)
            if limit > 0:
                cursor = cursor.limit(limit)
            documents = []
            async for doc in cursor:
                if "_id" in doc:
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
                if config.HIPAA_ANONYMIZE:
                    self._anonymize_document(doc)
                # Convert datetime fields to ISO strings (generic for all entities)
                for field in ["appointment_time", "date_of_birth", "created_at", "updated_at", "timestamp"]:
                    if field in doc and isinstance(doc[field], datetime):
                        doc[field] = doc[field].isoformat()
                documents.append(doc)
            logger.info(f"Found {len(documents)} documents in {collection_name}")
            return documents
        except PyMongoError as e:
            logger.error(f"Error finding documents in {collection_name}: {e}")
            raise
    
    async def find_one_document(
        self, 
        collection_name: str, 
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find single document."""
        try:
            collection = await self.get_collection(collection_name)
            doc = await collection.find_one(query)
            if doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                if config.HIPAA_ANONYMIZE:
                    self._anonymize_document(doc)
                # Convert datetime fields to ISO strings
                for field in ["appointment_time", "date_of_birth", "created_at", "updated_at", "timestamp"]:
                    if field in doc and isinstance(doc[field], datetime):
                        doc[field] = doc[field].isoformat()
            return doc
        except PyMongoError as e:
            logger.error(f"Error finding document in {collection_name}: {e}")
            raise
    
    async def update_documents(
        self, 
        collection_name: str, 
        query: Dict[str, Any], 
        update: Dict[str, Any],
        upsert: bool = False
    ) -> int:
        """Update documents matching query and return modified count."""
        try:
            collection = await self.get_collection(collection_name)
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update["$set"] = {"updated_at": datetime.utcnow()}
            result = await collection.update_many(query, update, upsert=upsert)
            logger.info(f"Updated {result.modified_count} documents in {collection_name}")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating documents in {collection_name}: {e}")
            raise
    
    async def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete documents matching query and return deleted count."""
        try:
            collection = await self.get_collection(collection_name)
            result = await collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents in {collection_name}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting documents in {collection_name}: {e}")
            raise
    
    async def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline."""
        try:
            collection = await self.get_collection(collection_name)
            cursor = collection.aggregate(pipeline)
            results = []
            async for doc in cursor:
                if "_id" in doc and doc["_id"] is not None:
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
                results.append(doc)
            logger.info(f"Aggregation returned {len(results)} results in {collection_name}")
            return results
        except PyMongoError as e:
            logger.error(f"Error running aggregation in {collection_name}: {e}")
            raise
    
    async def create_index(self, collection_name: str, index_spec: Dict[str, Any], **kwargs):
        """Create index on collection."""
        try:
            collection = await self.get_collection(collection_name)
            await collection.create_index(list(index_spec.items()), **kwargs)
            logger.info(f"Created index on {collection_name}: {index_spec}")
        except PyMongoError as e:
            logger.error(f"Error creating index in {collection_name}: {e}")
            raise
    
    async def close_connection(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")
    
    # Alias for consistency with previous code
    close = close_connection
    
    def _anonymize_document(self, document: Dict[str, Any]):
        """Anonymize sensitive fields for HIPAA compliance."""
        sensitive_fields = [
            'patient_name', 'patient_id', 'insurance_member_id', 
            'patient_phone_number', 'address', 'date_of_birth'
        ]
        for field in sensitive_fields:
            if field in document:
                if field == 'date_of_birth':
                    if isinstance(document[field], str) and len(document[field]) >= 4:
                        document[field] = document[field][:4] + "-XX-XX"
                    else:
                        document[field] = "XXXX-XX-XX"
                else:
                    document[field] = "REDACTED"
    
    # Domain-Specific Methods: Conversations (from previous merge)
    
    @staticmethod
    def validate_authorization_data(extracted_data: dict) -> dict:
        """Validate and clean extracted authorization data."""
        valid_statuses = ['approved', 'denied', 'pending', 'under_review', 'requires_documentation']
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
        auth_number = extracted_data.get("auth_number")
        if auth_number:
            auth_number = auth_number.upper().strip()
            for prefix in ['AUTH-', 'AUTH_', 'AUTH ']:
                if auth_number.startswith(prefix):
                    auth_number = auth_number[len(prefix):]
                    break
            else:
                if auth_number.startswith('AUTH'):
                    auth_number = auth_number[4:]
        patient_id = extracted_data.get("patient_id")
        if patient_id:
            patient_id = patient_id.strip().upper()
        procedure_code = extracted_data.get("procedure_code")
        if procedure_code:
            procedure_code = procedure_code.strip().upper()
            for prefix in ['CPT-', 'CPT_', 'CPT ']:
                if procedure_code.startswith(prefix):
                    procedure_code = procedure_code[len(prefix):]
                    break
            else:
                if procedure_code.startswith('CPT'):
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
        self,
        call_sid: str,
        user_input: str,
        bot_response: str,
        extracted_data: dict,
        escalated: bool = False
    ) -> str:
        """Log conversation data to MongoDB. Returns document ID as string."""
        validated_data = self.validate_authorization_data(extracted_data)
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
            return await self.insert_document("conversations", document)
        except Exception as e:
            logger.error(f"Failed to log conversation for call {call_sid}: {str(e)}", 
                         extra={"call_sid": call_sid})
            raise
    
    async def get_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Retrieve conversation logs from MongoDB."""
        try:
            return await self.find_documents(
                "conversations",
                query={},
                limit=limit,
                skip=skip,
                sort=[("timestamp", -1)]
            )
        except Exception as e:
            logger.error(f"Failed to retrieve conversation logs: {str(e)}")
            raise
    
    async def get_conversation_by_call_sid(self, call_sid: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific call SID."""
        try:
            return await self.find_documents(
                "conversations",
                query={"call_sid": call_sid},
                sort=[("timestamp", 1)]
            )
        except Exception as e:
            logger.error(f"Failed to get conversations for call {call_sid}: {str(e)}")
            raise
    
    async def update_conversation(self, doc_id: str, updates: Dict[str, Any]) -> int:
        """Update a conversation document."""
        try:
            return await self.update_documents(
                "conversations",
                query={"_id": ObjectId(doc_id)},
                update={"$set": updates}
            )
        except Exception as e:
            logger.error(f"Failed to update conversation {doc_id}: {str(e)}")
            raise
    
    async def delete_conversation(self, doc_id: str) -> int:
        """Delete a conversation document."""
        try:
            return await self.delete_documents(
                "conversations",
                query={"_id": ObjectId(doc_id)}
            )
        except Exception as e:
            logger.error(f"Failed to delete conversation {doc_id}: {str(e)}")
            raise
    
    async def get_conversation_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get conversation statistics for the last N days."""
        try:
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
            result = await self.aggregate("conversations", pipeline)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {str(e)}")
            raise
    
    # Domain-Specific Methods: Patients (integrated from patients.py)
    
    async def patient_create(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record. Returns document ID as string."""
        try:
            # Add metadata
            patient_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "prior_auth_status": patient_data.get("prior_auth_status", "Pending")
            })
            return await self.insert_document("patients", patient_data)
        except Exception as e:
            logger.error(f"Failed to create patient: {str(e)}")
            raise
    
    async def patient_get_all(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Retrieve all patients from the database."""
        try:
            return await self.find_documents(
                "patients",
                query={},
                limit=limit,
                skip=skip,
                sort=[("appointment_time", 1)]
            )
        except Exception as e:
            logger.error(f"Failed to retrieve patients: {str(e)}")
            raise
    
    async def patient_get_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific patient by ID."""
        try:
            return await self.find_one_document(
                "patients",
                query={"_id": ObjectId(patient_id)}
            )
        except Exception as e:
            logger.error(f"Failed to get patient {patient_id}: {str(e)}")
            raise
    
    async def patient_get_by_member_id(self, member_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by insurance member ID."""
        try:
            return await self.find_one_document(
                "patients",
                query={"insurance_member_id": member_id}
            )
        except Exception as e:
            logger.error(f"Failed to get patient by member ID {member_id}: {str(e)}")
            raise
    
    async def patient_update_auth_status(self, patient_id: str, status: str, auth_number: str = None) -> int:
        """Update the prior authorization status for a patient."""
        try:
            updates = {
                "prior_auth_status": status
            }
            if auth_number:
                updates["auth_number"] = auth_number
            return await self.update_documents(
                "patients",
                query={"_id": ObjectId(patient_id)},
                update={"$set": updates}
            )
        except Exception as e:
            logger.error(f"Failed to update patient auth status for {patient_id}: {str(e)}")
            raise
    
    async def patient_get_pending_authorizations(self) -> List[Dict[str, Any]]:
        """Get all patients with pending prior authorization status."""
        try:
            return await self.find_documents(
                "patients",
                query={"prior_auth_status": "Pending"},
                sort=[("appointment_time", 1)]
            )
        except Exception as e:
            logger.error(f"Failed to get pending authorizations: {str(e)}")
            raise
    
    async def patient_get_by_provider(self, provider_npi: str) -> List[Dict[str, Any]]:
        """Get all patients for a specific provider."""
        try:
            return await self.find_documents(
                "patients",
                query={"provider_npi": provider_npi},
                sort=[("appointment_time", 1)]
            )
        except Exception as e:
            logger.error(f"Failed to get patients for provider {provider_npi}: {str(e)}")
            raise
    
    async def patient_delete(self, patient_id: str) -> int:
        """Delete a patient record."""
        try:
            return await self.delete_documents(
                "patients",
                query={"_id": ObjectId(patient_id)}
            )
        except Exception as e:
            logger.error(f"Failed to delete patient {patient_id}: {str(e)}")
            raise

# Global connector instance
mongo_connector = AsyncMongoConnector()