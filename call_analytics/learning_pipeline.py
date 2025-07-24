import logging
import asyncio
from typing import Dict, Any, List, Optional
import datetime
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import nest_asyncio

nest_asyncio.apply()

from shared.config import config
from shared.providers.illm_provider import get_llm
from langchain_core.messages import HumanMessage

from .mongo_connector import AsyncMongoConnector
from .vector_index import AsyncVectorIndex

logger = logging.getLogger(__name__)

class PatternExtractor:
    """
    Extracts patterns using LLM: winning phrases, objection handling, success patterns.
    Analyzes transcripts to identify repeatable learnings.
    """

    def __init__(self):
        self.llm = get_llm(model="gpt-4", temperature=0.3, max_tokens=1000)
        logger.info("PatternExtractor initialized")

    def extract_patterns(self, transcript: str, analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract patterns from transcript and optional analysis.

        :param transcript: Full transcript text.
        :param analysis: Optional analysis dict.
        :return: List of patterns {'type': str, 'phrase': str, 'context': str, 'outcome': str}.
        """
        prompt = self._build_extraction_prompt(transcript, analysis)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            patterns = self._parse_llm_response(response.content)
            logger.info(f"Extracted {len(patterns)} patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return []

    def _build_extraction_prompt(self, transcript: str, analysis: Optional[Dict[str, Any]]) -> str:
        analysis_str = json.dumps(analysis) if analysis else "None"
        return f"""
Analyze this healthcare prior authorization transcript for patterns:
- Winning phrases leading to success.
- Effective objection handling.
- Success/failure patterns (timing, phrasing).

Transcript: {transcript}
Analysis: {analysis_str}

Output ONLY JSON list: [{{"type": "winning_phrase", "phrase": "...", "context": "...", "outcome": "success"}}, ...]
Types: winning_phrase, objection_handling, success_pattern, failure_pattern.
"""

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().lstrip('```json').rstrip('```')
            patterns = json.loads(cleaned)
            if not isinstance(patterns, list) or not all('type' in p and 'phrase' in p and 'context' in p and 'outcome' in p for p in patterns):
                raise ValueError("Invalid pattern structure")
            return patterns
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Parse error: {e}")
            return []

class ScriptGenerator:
    """
    Generates scripts/responses using context-aware LLM prompts.
    Uses patterns and context for optimal responses.
    """

    def __init__(self):
        self.llm = get_llm(model="gpt-4", temperature=0.5, max_tokens=500)
        logger.info("ScriptGenerator initialized")

    def generate_script(self, context: str, patterns: Optional[List[Dict[str, Any]]] = None, scenario: str = "objection_handling") -> str:
        """
        Generate script for scenario.

        :param context: Conversation context.
        :param patterns: Optional patterns.
        :param scenario: Scenario type.
        :return: Generated script.
        """
        prompt = self._build_generation_prompt(context, patterns, scenario)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            script = self._parse_llm_response(response.content)
            logger.info("Script generated")
            return script
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            return ""

    def _build_generation_prompt(self, context: str, patterns: Optional[List[Dict[str, Any]]], scenario: str) -> str:
        patterns_str = "\n".join(f"- {p['phrase']}: {p['context']}" for p in patterns) if patterns else "None"
        return f"""
Generate professional script for healthcare prior authorization.

Scenario: {scenario}
Context: {context}
Patterns: {patterns_str}

Requirements: Empathetic, professional, concise (2-3 sentences), focus on patient care.
Output only the script text.
"""

    def _parse_llm_response(self, response: str) -> str:
        script = response.strip().strip('"').lstrip("Rep:").strip()
        return script

class MemoryManager:
    """
    Manages agent memory in MongoDB: generation, storage, retrieval, update, forget.
    Handles types: short_term, long_term, episodic, procedural, conversational, entity, workflow.
    Uses RAG via VectorIndex.
    """

    def __init__(self):
        self.connector = AsyncMongoConnector()
        self.vector_index = AsyncVectorIndex()
        self.extractor = PatternExtractor()
        self.memory_types = ["short_term", "long_term", "episodic", "procedural", "conversational", "entity", "workflow"]
        for mtype in self.memory_types:
            asyncio.run(self.vector_index.create_vector_index(mtype, dimensions=384))
            logger.info(f"Created vector index for {mtype}")
        logger.info("MemoryManager initialized")

    def generate_memory(self, data: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        """
        Generate memory from data.

        :param data: Input data.
        :param memory_type: Memory type.
        :return: Memory dict.
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory_type: {memory_type}")
        transcript = data.get("transcript", "")
        patterns = self.extractor.extract_patterns(transcript, data.get("analysis"))
        memory = {
            "type": memory_type,
            "content": json.dumps(patterns),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "relevance_score": 1.0,
            "source_id": data.get("source_id", ""),
            "text": " ".join(p["phrase"] for p in patterns)
        }
        logger.info(f"Generated {memory_type} memory")
        return memory

    def store_memory(self, memory: Dict[str, Any]):
        """
        Store memory with embedding.

        :param memory: Memory dict.
        """
        mtype = memory["type"]
        self.vector_index.insert_with_embedding(mtype, memory, "text")

    def retrieve_memory(self, query: str, memory_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories via vector search.

        :param query: Query text.
        :param memory_type: Type.
        :param top_k: Results count.
        :return: Memories list.
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory_type: {memory_type}")
        results = self.vector_index.vector_search(memory_type, query, top_k)
        memories = [r["document"] for r in results]
        logger.info(f"Retrieved {len(memories)} from {memory_type}")
        return memories

    def integrate_memory(self, new_memory: Dict[str, Any], existing_id: Optional[str] = None):
        """
        Integrate: store new or update.

        :param new_memory: New memory.
        :param existing_id: ID to update.
        """
        if existing_id:
            self.connector.update_document(new_memory["type"], {"_id": existing_id}, {"$set": new_memory})
            logger.info(f"Updated ID: {existing_id}")
        else:
            self.store_memory(new_memory)

    def update_memory(self, memory_type: str, query: Dict[str, Any], updates: Dict[str, Any]):
        """
        Update matching memories.

        :param memory_type: Type.
        :param query: Filter.
        :param updates: Updates.
        """
        self.connector.update_document(memory_type, query, {"$set": updates})

    def forget_memory(self, memory_type: str, threshold_days: int = 30, min_relevance: float = 0.5):
        """
        Delete old/low-relevance memories.

        :param memory_type: Type.
        :param threshold_days: Age threshold.
        :param min_relevance: Score threshold.
        """
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=threshold_days)).isoformat()
        query = {"$or": [{"timestamp": {"$lt": cutoff}}, {"relevance_score": {"$lt": min_relevance}}]}
        deleted = self.connector.delete_document(memory_type, query)
        logger.info(f"Forgot {deleted} from {memory_type}")

    def close(self):
        """
        Close connections.
        """
        self.connector.close_connection()
        self.vector_index.close()

class TrainingPipeline:
    """
    Prepares training data: collect, annotate, format, split/export.
    """

    def __init__(self):
        self.connector = AsyncMongoConnector()
        self.extractor = PatternExtractor()
        self.generator = ScriptGenerator()
        logger.info("TrainingPipeline initialized")

    def collect_data(self, collection_name: str = "transcripts", query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Collect from MongoDB.

        :param collection_name: Collection.
        :param query: Filter.
        :return: Documents list.
        """
        data = self.connector.find_documents(collection_name, query or {})
        logger.info(f"Collected {len(data)} from {collection_name}")
        return data

    def annotate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate with patterns/scripts/labels.

        :param data: Raw data.
        :return: Annotated data.
        """
        annotated = []
        for item in data:
            transcript = item.get("transcript", "")
            if not transcript:
                continue
            patterns = self.extractor.extract_patterns(transcript)
            item["patterns"] = patterns
            context = " ".join(p["phrase"] for p in patterns if p["type"] == "objection_handling")
            if context:
                item["generated_script"] = self.generator.generate_script(context)
            item["label"] = 1 if item.get("outcome") == "success" else 0
            annotated.append(item)
        logger.info(f"Annotated {len(annotated)}")
        return annotated

    def format_data(self, annotated: List[Dict[str, Any]], format_type: str = "fine_tune") -> pd.DataFrame:
        """
        Format to DataFrame.

        :param annotated: Annotated.
        :param format_type: 'fine_tune' or 'rag'.
        :return: DataFrame.
        """
        if format_type == "fine_tune":
            rows = [{"prompt": f"Transcript: {item['transcript']}\nPatterns: {json.dumps(item['patterns'])}", "response": item.get("generated_script", ""), "label": item["label"]} for item in annotated]
        elif format_type == "rag":
            rows = [{"context": item["transcript"], "patterns": item["patterns"]} for item in annotated]
        else:
            raise ValueError(f"Unknown format: {format_type}")
        df = pd.DataFrame(rows)
        logger.info(f"Formatted {len(df)} rows")
        return df

    def split_and_export(self, df: pd.DataFrame, export_path: str = "training_data.json", test_size: float = 0.2):
        """
        Split/export data.

        :param df: DataFrame.
        :param export_path: Path.
        :param test_size: Split ratio.
        """
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        ext = Path(export_path).suffix
        if ext == ".json":
            train_df.to_json(export_path.replace(ext, f"_train{ext}"), orient="records", indent=4)
            test_df.to_json(export_path.replace(ext, f"_test{ext}"), orient="records", indent=4)
        elif ext == ".csv":
            train_df.to_csv(export_path.replace(ext, f"_train{ext}"), index=False)
            test_df.to_csv(export_path.replace(ext, f"_test{ext}"), index=False)
        else:
            raise ValueError("Unsupported format")
        logger.info(f"Exported to {export_path}")

    def run_pipeline(self, collection_name: str = "transcripts"):
        """
        Full pipeline run.
        """
        data = self.collect_data(collection_name)
        annotated = self.annotate_data(data)
        df = self.format_data(annotated)
        self.split_and_export(df)