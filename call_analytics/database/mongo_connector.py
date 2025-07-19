# Enhanced async MongoDB connector for Alfons analytics system
# Replaces sync connector with async motor implementation

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, PyMongoError
from bson.objectid import ObjectId
from datetime import datetime

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

class AsyncMongoConnector:
    """
    Async MongoDB connector for analytics operations.
    Handles connection, CRUD operations, and aggregations with proper error handling.
    """
    
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self):
        """Initialize async MongoDB connection"""
        try:
            self._client = AsyncIOMotorClient(settings.MONGO_URI)
            # Test connection
            await self._client.admin.command('ping')
            self._db = self._client[settings.MONGO_DB_NAME]
            logger.info(f"Connected to MongoDB database: {settings.MONGO_DB_NAME}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @property
    async def client(self) -> AsyncIOMotorClient:
        if self._client is None:
            await self.connect()
        return self._client
    
    @property
    async def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            await self.connect()
        return self._db
    
    async def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get async collection reference"""
        db = await self.db
        return db[collection_name]
    
    async def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert single document and return ID as string"""
        try:
            collection = await self.get_collection(collection_name)
            
            # Add timestamp if not present
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            
            result = await collection.insert_one(document)
            logger.info(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error inserting document: {e}")
            raise
    
    async def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents and return IDs as strings"""
        try:
            collection = await self.get_collection(collection_name)
            
            # Add timestamps
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow()
            
            result = await collection.insert_many(documents)
            inserted_ids = [str(id) for id in result.inserted_ids]
            logger.info(f"Inserted {len(inserted_ids)} documents")
            return inserted_ids
        except PyMongoError as e:
            logger.error(f"Error inserting multiple documents: {e}")
            raise
    
    async def find_documents(
        self, 
        collection_name: str, 
        query: Dict[str, Any] = {}, 
        limit: int = 0,
        skip: int = 0,
        sort: Optional[List[tuple]] = None
    ) -> List[Dict[str, Any]]:
        """Find documents with optional pagination and sorting"""
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
                # Convert ObjectId to string
                if "_id" in doc:
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
                
                # Apply anonymization if enabled
                if settings.HIPAA_ANONYMIZE:
                    self._anonymize_document(doc)
                
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} documents")
            return documents
        except PyMongoError as e:
            logger.error(f"Error finding documents: {e}")
            raise
    
    async def find_one_document(
        self, 
        collection_name: str, 
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find single document"""
        try:
            collection = await self.get_collection(collection_name)
            doc = await collection.find_one(query)
            
            if doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                
                if settings.HIPAA_ANONYMIZE:
                    self._anonymize_document(doc)
            
            return doc
        except PyMongoError as e:
            logger.error(f"Error finding document: {e}")
            raise
    
    async def update_document(
        self, 
        collection_name: str, 
        query: Dict[str, Any], 
        update: Dict[str, Any],
        upsert: bool = False
    ) -> int:
        """Update documents matching query"""
        try:
            collection = await self.get_collection(collection_name)
            
            # Add updated timestamp
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update["$set"] = {"updated_at": datetime.utcnow()}
            
            result = await collection.update_many(query, update, upsert=upsert)
            logger.info(f"Updated {result.modified_count} documents")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    async def delete_document(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete documents matching query"""
        try:
            collection = await self.get_collection(collection_name)
            result = await collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    async def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline"""
        try:
            collection = await self.get_collection(collection_name)
            cursor = collection.aggregate(pipeline)
            results = []
            
            async for doc in cursor:
                if "_id" in doc and doc["_id"] is not None:
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
                results.append(doc)
            
            logger.info(f"Aggregation returned {len(results)} results")
            return results
        except PyMongoError as e:
            logger.error(f"Error running aggregation: {e}")
            raise
    
    async def create_index(self, collection_name: str, index_spec: Dict[str, Any], **kwargs):
        """Create index on collection"""
        try:
            collection = await self.get_collection(collection_name)
            await collection.create_index(list(index_spec.items()), **kwargs)
            logger.info(f"Created index on {collection_name}: {index_spec}")
        except PyMongoError as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def close_connection(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")
    
    def _anonymize_document(self, document: Dict[str, Any]):
        """Anonymize sensitive fields for HIPAA compliance"""
        sensitive_fields = [
            'patient_name', 'patient_id', 'insurance_member_id', 
            'patient_phone_number', 'address', 'date_of_birth'
        ]
        
        for field in sensitive_fields:
            if field in document:
                if field == 'date_of_birth':
                    # Keep year, anonymize month/day
                    if isinstance(document[field], str) and len(document[field]) >= 4:
                        document[field] = document[field][:4] + "-XX-XX"
                    else:
                        document[field] = "XXXX-XX-XX"
                else:
                    document[field] = "REDACTED"

# Global connector instance
mongo_connector = AsyncMongoConnector()