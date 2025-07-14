import logging
from typing import Dict, Any, List, Optional
import datetime
import json

from call_analytics.config.settings import settings
from ..database.mongo_connector import MongoConnector
from ..database.vector_index import VectorIndex
from .pattern_extractor import PatternExtractor

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages agent memory in MongoDB: generation, storage, retrieval, integration, updating, forgetting.
    Handles memory types: short-term/long-term, episodic/procedural, conversational/entity/workflow.
    Uses RAG for retrieval; embeddings via VectorIndex.
    Inspired by Richmond Alake: mimic human brain, flexible schemas.
    Forgetting: time-based or relevance-based pruning.
    """
    
    def __init__(self):
        """
        Initialize memory manager with connectors.
        """
        self.connector = MongoConnector()
        self.vector_index = VectorIndex()
        self.extractor = PatternExtractor()  # For generating memories from data
        self.memory_types = [
            "short_term", "long_term", "episodic", "procedural",
            "conversational", "entity", "workflow"
        ]
        for mtype in self.memory_types:
            # Ensure collections exist; create vector index if needed
            self.vector_index.create_vector_index(mtype, dimensions=384)
        
        logger.info("MemoryManager initialized")
    
    def generate_memory(self, data: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        """
        Generate a memory entry from data (e.g., transcript, patterns).
        
        :param data: Input data (e.g., {'transcript': str, 'analysis': dict}).
        :param memory_type: One of memory_types.
        :return: Generated memory dict.
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory_type: {memory_type}")
        
        transcript = data.get("transcript", "")
        patterns = self.extractor.extract_patterns(transcript, data.get("analysis"))
        
        memory = {
            "type": memory_type,
            "content": json.dumps(patterns),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "relevance_score": 1.0,  # Initial score; update later
            "source_id": data.get("source_id", "")
        }
        
        # Add text for embedding
        memory["text"] = " ".join([p["phrase"] for p in patterns])
        
        logger.info(f"Generated memory of type {memory_type}")
        return memory
    
    def store_memory(self, memory: Dict[str, Any]):
        """
        Store memory in appropriate collection with embedding.
        
        :param memory: Memory dict.
        """
        mtype = memory["type"]
        self.vector_index.insert_with_embedding(mtype, memory, "text")
    
    def retrieve_memory(self, query: str, memory_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories using RAG (vector search).
        
        :param query: Query text.
        :param memory_type: Memory type to search.
        :param top_k: Number of results.
        :return: List of matching memories with scores.
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory_type: {memory_type}")
        
        results = self.vector_index.vector_search(memory_type, query, top_k)
        memories = [r["document"] for r in results]
        logger.info(f"Retrieved {len(memories)} memories from {memory_type}")
        return memories
    
    def integrate_memory(self, new_memory: Dict[str, Any], existing_id: Optional[str] = None):
        """
        Integrate new memory: store new or update existing.
        
        :param new_memory: New memory dict.
        :param existing_id: Optional ID to update.
        """
        if existing_id:
            self.connector.update_document(new_memory["type"], {"_id": existing_id}, {"$set": new_memory})
            logger.info(f"Updated memory ID: {existing_id}")
        else:
            self.store_memory(new_memory)
    
    def update_memory(self, memory_type: str, query: Dict[str, Any], updates: Dict[str, Any]):
        """
        Update memories matching query.
        
        :param memory_type: Memory type.
        :param query: Filter query.
        :param updates: Update dict.
        """
        self.connector.update_document(memory_type, query, {"$set": updates})
    
    def forget_memory(self, memory_type: str, threshold_days: int = 30, min_relevance: float = 0.5):
        """
        Forget (delete) old or low-relevance memories.
        
        :param memory_type: Memory type.
        :param threshold_days: Delete if older than this.
        :param min_relevance: Delete if relevance below this.
        """
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=threshold_days)).isoformat()
        query = {
            "$or": [
                {"timestamp": {"$lt": cutoff}},
                {"relevance_score": {"$lt": min_relevance}}
            ]
        }
        deleted = self.connector.delete_document(memory_type, query)
        logger.info(f"Forgot {deleted} memories from {memory_type}")
    
    def close(self):
        """
        Close connections.
        """
        self.connector.close_connection()
        self.vector_index.close()

# Example usage (for testing)
if __name__ == "__main__":
    manager = MemoryManager()
    try:
        mock_data = {"transcript": "Sample call transcript.", "analysis": {}}
        memory = manager.generate_memory(mock_data, "episodic")
        manager.store_memory(memory)
        
        retrieved = manager.retrieve_memory("Sample", "episodic", top_k=1)
        print(retrieved)
        
        manager.forget_memory("episodic", threshold_days=0)  # Clean up
    except Exception as e:
        print(f"Error: {e}")
    finally:
        manager.close()