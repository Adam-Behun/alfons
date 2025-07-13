import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from config.settings import settings
from .mongo_connector import MongoConnector  # Reuse connector for consistency

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    Handles embeddings for RAG using sentence-transformers.
    Manages embedding generation, indexing in MongoDB, and vector search.
    Assumes MongoDB Vector Search is set up (e.g., Atlas or local with vector support).
    For MVP, stores embeddings in documents and uses cosine similarity queries.
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize the embedding model and MongoDB connection.
        
        :param model_name: Name of the sentence-transformers model.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self.connector = MongoConnector()
        self.client = self.connector.client
        self.db = self.connector.db
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        :param texts: List of text strings to embed.
        :return: List of embedding vectors (as lists for MongoDB storage).
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            embeddings_list = embeddings.tolist()
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def create_vector_index(self, collection_name: str, field_name: str = "embedding", dimensions: int = 384):
        """
        Create a vector search index in MongoDB (requires MongoDB Atlas or compatible).
        
        :param collection_name: Name of the collection.
        :param field_name: Field to index (e.g., 'embedding').
        :param dimensions: Dimensionality of the embeddings.
        """
        try:
            collection = self.db[collection_name]
            index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": field_name,
                            "numDimensions": dimensions,
                            "similarity": "cosine"
                        }
                    ]
                },
                name="vector_index"
            )
            collection.create_search_index(index_model)
            logger.info(f"Created vector index on {collection_name}.{field_name}")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    def insert_with_embedding(self, collection_name: str, document: Dict[str, Any], text_field: str, embedding_field: str = "embedding"):
        """
        Insert a document with generated embedding.
        
        :param collection_name: Name of the collection.
        :param document: Document to insert.
        :param text_field: Field containing text to embed.
        :param embedding_field: Field to store the embedding.
        :return: Inserted document ID.
        """
        text = document.get(text_field)
        if not text:
            raise ValueError("Text field is empty or missing")
        
        embedding = self.generate_embeddings([text])[0]
        document[embedding_field] = embedding
        return self.connector.insert_document(collection_name, document)
    
    def vector_search(self, collection_name: str, query_text: str, top_k: int = 5, embedding_field: str = "embedding") -> List[Dict[str, Any]]:
        """
        Perform vector search using cosine similarity (manual fallback if no native vector search).
        
        :param collection_name: Name of the collection.
        :param query_text: Query text to embed and search.
        :param top_k: Number of top results to return.
        :param embedding_field: Field containing embeddings.
        :return: List of matching documents with scores.
        """
        query_embedding = self.generate_embeddings([query_text])[0]
        collection = self.db[collection_name]
        
        # Manual cosine similarity (for environments without native vector search)
        pipeline = [
            {"$addFields": {"similarity": {"$meta": "searchScore"}}},  # Placeholder; actually compute manually
            {"$project": {"_id": 1, "text": 1, "score": {"$meta": "searchScore"}}},
            {"$limit": top_k}
        ]
        
        # For simplicity, fetch all and compute (not scalable; use for MVP)
        docs = list(collection.find({embedding_field: {"$exists": True}}))
        similarities = []
        for doc in docs:
            doc_emb = np.array(doc[embedding_field])
            query_emb = np.array(query_embedding)
            sim = np.dot(doc_emb, query_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(query_emb))
            similarities.append((doc, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [{"document": s[0], "score": s[1]} for s in similarities[:top_k]]
        logger.info(f"Vector search returned {len(results)} results")
        return results
    
    def close(self):
        """
        Close the MongoDB connection.
        """
        self.connector.close_connection()

# Example usage (for testing)
if __name__ == "__main__":
    vector_index = VectorIndex()
    try:
        # Test embedding
        emb = vector_index.generate_embeddings(["Test sentence"])
        print(f"Embedding length: {len(emb[0])}")
        
        # Test insert
        test_doc = {"text": "Sample text for embedding"}
        inserted_id = vector_index.insert_with_embedding("test_vectors", test_doc, "text")
        print(f"Inserted ID: {inserted_id}")
        
        # Test search
        results = vector_index.vector_search("test_vectors", "Sample text", top_k=1)
        print(f"Search results: {results}")
    finally:
        vector_index.close()