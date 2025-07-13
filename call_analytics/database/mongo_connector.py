import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, PyMongoError
from bson.objectid import ObjectId
from typing import Any, Dict, List, Optional

from config.settings import settings  # Assuming config is in the parent directory or adjusted import path

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL)

class MongoConnector:
    """
    MongoDB connector for CRUD operations in the Alfons system.
    Handles connection, document insertion, querying, updating, and deletion.
    Supports memory management with flexible schemas for different memory types.
    Ensures error handling and logging.
    """
    
    def __init__(self):
        """
        Initialize the MongoDB client and database connection.
        """
        try:
            self.client = MongoClient(settings.MONGO_URI)
            self.db = self.client[settings.MONGO_DB_NAME]
            logger.info(f"Connected to MongoDB database: {settings.MONGO_DB_NAME}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a reference to a MongoDB collection.
        
        :param collection_name: Name of the collection.
        :return: MongoDB Collection object.
        """
        return self.db[collection_name]
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document into a collection.
        
        :param collection_name: Name of the collection.
        :param document: Dictionary representing the document.
        :return: Inserted document ID as string.
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            logger.info(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error inserting document: {e}")
            raise
    
    def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents into a collection.
        
        :param collection_name: Name of the collection.
        :param documents: List of dictionaries representing documents.
        :return: List of inserted document IDs as strings.
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            inserted_ids = [str(id) for id in result.inserted_ids]
            logger.info(f"Inserted {len(inserted_ids)} documents")
            return inserted_ids
        except PyMongoError as e:
            logger.error(f"Error inserting multiple documents: {e}")
            raise
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = {}, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Find documents matching a query.
        
        :param collection_name: Name of the collection.
        :param query: Query filter dictionary.
        :param limit: Maximum number of documents to return (0 for no limit).
        :return: List of matching documents.
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query).limit(limit)
            documents = list(cursor)
            # Anonymize sensitive data if enabled
            if settings.HIPAA_ANONYMIZE:
                for doc in documents:
                    self._anonymize_document(doc)
            logger.info(f"Found {len(documents)} documents")
            return documents
        except PyMongoError as e:
            logger.error(f"Error finding documents: {e}")
            raise
    
    def update_document(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """
        Update documents matching a query.
        
        :param collection_name: Name of the collection.
        :param query: Query filter dictionary.
        :param update: Update operations dictionary.
        :return: Number of modified documents.
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_many(query, update)
            logger.info(f"Updated {result.modified_count} documents")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def delete_document(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete documents matching a query.
        
        :param collection_name: Name of the collection.
        :param query: Query filter dictionary.
        :return: Number of deleted documents.
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def close_connection(self):
        """
        Close the MongoDB client connection.
        """
        self.client.close()
        logger.info("MongoDB connection closed")
    
    def _anonymize_document(self, document: Dict[str, Any]):
        """
        Anonymize sensitive fields in a document for HIPAA-like privacy.
        This is a placeholder; implement specific anonymization logic as needed.
        
        :param document: Document to anonymize (modified in-place).
        """
        # Example: Redact patient names, IDs, etc.
        if 'patient_name' in document:
            document['patient_name'] = 'REDACTED'
        if 'patient_id' in document:
            document['patient_id'] = 'REDACTED'
        # Add more fields as per requirements

# Example usage (for testing purposes; remove in production)
if __name__ == "__main__":
    connector = MongoConnector()
    try:
        # Test insert
        test_doc = {"type": "test", "data": "sample"}
        inserted_id = connector.insert_document("test_collection", test_doc)
        print(f"Inserted ID: {inserted_id}")
        
        # Test find
        found = connector.find_documents("test_collection", {"_id": ObjectId(inserted_id)})
        print(f"Found: {found}")
        
        # Test update
        connector.update_document("test_collection", {"_id": ObjectId(inserted_id)}, {"$set": {"data": "updated"}})
        
        # Test delete
        connector.delete_document("test_collection", {"_id": ObjectId(inserted_id)})
    finally:
        connector.close_connection()