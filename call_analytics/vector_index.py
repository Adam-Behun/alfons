# Enhanced async vector search for MongoDB with embeddings
# Supports both MongoDB Atlas Vector Search and fallback similarity

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from motor.motor_asyncio import AsyncIOMotorCollection
import asyncio

from shared.config import config
from shared.logging import get_logger
from .mongo_connector import AsyncMongoConnector

logger = get_logger(__name__)

class AsyncVectorIndex:
    """
    Async vector search implementation for MongoDB.
    Supports embeddings generation, storage, and similarity search.
    Uses MongoDB Atlas Vector Search when available, fallback to manual cosine similarity.
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.connector = AsyncMongoConnector()
        
    async def initialize(self):
        """Initialize embedding model and MongoDB connection"""
        try:
            # Initialize model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            logger.info(f"Loaded embedding model: {self.model_name}")
            
            await self.connector.connect()
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        if not self.model:
            await self.initialize()
        
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )
            embeddings_list = embeddings.tolist()
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def create_vector_index(
        self, 
        collection_name: str, 
        field_name: str = "embedding", 
        dimensions: int = 384
    ):
        """Create vector search index (requires MongoDB Atlas)"""
        try:
            collection = await self.connector.get_collection(collection_name)
            
            # MongoDB Atlas Vector Search index definition
            index_definition = {
                "fields": [{
                    "type": "vector",
                    "path": field_name,
                    "numDimensions": dimensions,
                    "similarity": "cosine"
                }]
            }
            
            # Note: create_search_index is Atlas-specific
            # For local MongoDB, this will create a regular index
            try:
                await collection.create_search_index({
                    "definition": index_definition,
                    "name": f"{field_name}_vector_index"
                })
                logger.info(f"Created vector search index on {collection_name}.{field_name}")
            except Exception as atlas_error:
                # Fallback to regular index for local MongoDB
                await collection.create_index([(field_name, "2dsphere")])
                logger.info(f"Created fallback index on {collection_name}.{field_name}")
                
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    async def insert_with_embedding(
        self, 
        collection_name: str, 
        document: Dict[str, Any], 
        text_field: str, 
        embedding_field: str = "embedding"
    ) -> str:
        """Insert document with generated embedding"""
        text = document.get(text_field)
        if not text:
            raise ValueError(f"Text field '{text_field}' is empty or missing")
        
        # Generate embedding
        embeddings = await self.generate_embeddings([text])
        document[embedding_field] = embeddings[0]
        
        # Insert document
        doc_id = await self.connector.insert_document(collection_name, document)
        logger.info(f"Inserted document with embedding: {doc_id}")
        return doc_id
    
    async def vector_search(
        self, 
        collection_name: str, 
        query_text: str, 
        top_k: int = 5, 
        embedding_field: str = "embedding",
        filter_query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        Uses MongoDB Atlas Vector Search if available, falls back to manual computation
        """
        # Generate query embedding
        query_embeddings = await self.generate_embeddings([query_text])
        query_embedding = query_embeddings[0]
        
        collection = await self.connector.get_collection(collection_name)
        
        # Try MongoDB Atlas Vector Search first
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": f"{embedding_field}_vector_index",
                        "path": embedding_field,
                        "queryVector": query_embedding,
                        "numCandidates": top_k * 10,
                        "limit": top_k
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Add filter if provided
            if filter_query:
                pipeline.append({"$match": filter_query})
            results = []
           async for doc in collection.aggregate(pipeline):
               if "_id" in doc:
                   doc["id"] = str(doc["_id"])
                   del doc["_id"]
               results.append(doc)
           
           if results:
               logger.info(f"Atlas vector search returned {len(results)} results")
               return results
               
       except Exception as atlas_error:
           logger.warning(f"Atlas vector search failed, using fallback: {atlas_error}")
       
       # Fallback to manual cosine similarity
       return await self._manual_vector_search(
           collection, query_embedding, top_k, embedding_field, filter_query
       )
   
   async def _manual_vector_search(
       self,
       collection: AsyncIOMotorCollection,
       query_embedding: List[float],
       top_k: int,
       embedding_field: str,
       filter_query: Optional[Dict[str, Any]] = None
   ) -> List[Dict[str, Any]]:
       """Manual cosine similarity search for non-Atlas environments"""
       try:
           # Build query
           search_query = {embedding_field: {"$exists": True}}
           if filter_query:
               search_query.update(filter_query)
           
           # Fetch documents with embeddings
           docs = []
           async for doc in collection.find(search_query):
               docs.append(doc)
           
           # Compute similarities
           similarities = []
           query_emb = np.array(query_embedding)
           
           for doc in docs:
               try:
                   doc_emb = np.array(doc[embedding_field])
                   similarity = np.dot(doc_emb, query_emb) / (
                       np.linalg.norm(doc_emb) * np.linalg.norm(query_emb)
                   )
                   
                   # Convert ObjectId to string
                   if "_id" in doc:
                       doc["id"] = str(doc["_id"])
                       del doc["_id"]
                   
                   similarities.append({
                       "document": doc,
                       "score": float(similarity)
                   })
               except Exception as e:
                   logger.warning(f"Error computing similarity for document: {e}")
                   continue
           
           # Sort by similarity and return top k
           similarities.sort(key=lambda x: x["score"], reverse=True)
           results = similarities[:top_k]
           
           logger.info(f"Manual vector search returned {len(results)} results")
           return results
           
       except Exception as e:
           logger.error(f"Error in manual vector search: {e}")
           raise
   
   async def update_embedding(
       self,
       collection_name: str,
       doc_id: str,
       text: str,
       embedding_field: str = "embedding"
   ) -> int:
       """Update embedding for existing document"""
       embeddings = await self.generate_embeddings([text])
       
       result = await self.connector.update_document(
           collection_name,
           {"_id": ObjectId(doc_id)},
           {"$set": {embedding_field: embeddings[0]}}
       )
       
       logger.info(f"Updated embedding for document {doc_id}")
       return result
   
   async def bulk_generate_embeddings(
       self,
       collection_name: str,
       text_field: str,
       embedding_field: str = "embedding",
       batch_size: int = 100
   ):
       """Generate embeddings for all documents in collection that don't have them"""
       collection = await self.connector.get_collection(collection_name)
       
       # Find documents without embeddings
       query = {
           text_field: {"$exists": True, "$ne": ""},
           embedding_field: {"$exists": False}
       }
       
       total_processed = 0
       async for doc_batch in self._batch_cursor(collection.find(query), batch_size):
           texts = [doc[text_field] for doc in doc_batch if doc.get(text_field)]
           if not texts:
               continue
           
           # Generate embeddings
           embeddings = await self.generate_embeddings(texts)
           
           # Update documents
           for doc, embedding in zip(doc_batch, embeddings):
               await collection.update_one(
                   {"_id": doc["_id"]},
                   {"$set": {embedding_field: embedding}}
               )
           
           total_processed += len(doc_batch)
           logger.info(f"Processed {total_processed} documents for embeddings")
   
   async def _batch_cursor(self, cursor, batch_size: int):
       """Helper to batch cursor results"""
       batch = []
       async for doc in cursor:
           batch.append(doc)
           if len(batch) >= batch_size:
               yield batch
               batch = []
       if batch:
           yield batch
   
   async def get_similar_conversations(
       self,
       query_text: str,
       limit: int = 5,
       min_score: float = 0.7
   ) -> List[Dict[str, Any]]:
       """Find similar conversations for RAG context"""
       results = await self.vector_search(
           "conversations",
           query_text,
           top_k=limit,
           embedding_field="bot_response_embedding"
       )
       
       # Filter by minimum score
       filtered_results = [
           r for r in results 
           if r.get("score", 0) >= min_score
       ]
       
       return filtered_results
   
   async def close(self):
       """Close connections"""
       await self.connector.close_connection()

# Global vector index instance
vector_index = AsyncVectorIndex()