"""
Enhanced conversation management with state tracking, RAG integration, and OpenAI Realtime API support.
Handles both traditional text-based conversations and real-time speech interactions.
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from openai import AsyncOpenAI
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


class ConversationMode(Enum):
    """Conversation processing mode."""
    TEXT = "text"           # Traditional text-based processing
    REALTIME = "realtime"   # Real-time speech conversation
    HYBRID = "hybrid"       # Mixed text and speech


class ConversationState(Enum):
    """Prior authorization conversation states."""
    GREETING = "greeting"
    PATIENT_VERIFICATION = "patient_verification"
    INSURANCE_VERIFICATION = "insurance_verification"
    AUTH_REQUEST = "auth_request"
    MEDICAL_NECESSITY = "medical_necessity"
    OBJECTION_HANDLING = "objection_handling"
    INFORMATION_GATHERING = "information_gathering"
    DOCUMENTATION = "documentation"
    ESCALATION = "escalation"
    CLOSING = "closing"
    COMPLETED = "completed"


@dataclass
class ExtractedData:
    """Structured data extracted from conversation."""
    patient_id: Optional[str] = None
    procedure_code: Optional[str] = None
    diagnosis_code: Optional[str] = None
    insurance: Optional[str] = None
    auth_number: Optional[str] = None
    provider_npi: Optional[str] = None
    member_id: Optional[str] = None
    date_of_service: Optional[str] = None
    urgency_level: str = "standard"
    prior_auth_number: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update fields from dictionary."""
        for key, value in data.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class ConversationContext:
    """Complete conversation context and state."""
    conversation_id: str
    call_sid: Optional[str] = None
    mode: ConversationMode = ConversationMode.TEXT
    state: ConversationState = ConversationState.GREETING
    
    # Extracted data
    extracted_data: ExtractedData = None
    
    # Conversation history
    messages: List[Dict[str, Any]] = None
    turn_count: int = 0
    
    # State management
    escalated: bool = False
    escalation_reason: Optional[str] = None
    confidence_score: float = 1.0
    
    # Timing
    start_time: datetime = None
    last_activity: datetime = None
    
    # Context for RAG
    context_keywords: List[str] = None
    similar_cases: List[str] = None
    
    def __post_init__(self):
        if self.extracted_data is None:
            self.extracted_data = ExtractedData()
        if self.messages is None:
            self.messages = []
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.context_keywords is None:
            self.context_keywords = []
        if self.similar_cases is None:
            self.similar_cases = []


class RAGContextManager:
    """Manages retrieval-augmented generation for conversation enhancement."""
    
    def __init__(self, mongodb_client: AsyncIOMotorClient):
        """Initialize RAG context manager."""
        self.db = mongodb_client[config.DATABASE_NAME]
        self.embeddings_collection = self.db["conversation_embeddings"]
        self.knowledge_base = self.db["prior_auth_knowledge"]
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        # Cache for embeddings and frequent queries
        self.embedding_cache = {}
        self.knowledge_cache = {}
    
    async def get_relevant_context(
        self,
        query: str,
        conversation_state: ConversationState,
        extracted_data: ExtractedData,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for current conversation state.
        
        Args:
            query: Current conversation query
            conversation_state: Current state of conversation
            extracted_data: Currently extracted data
            max_results: Maximum number of context items to return
            
        Returns:
            Relevant context including patterns, objection responses, and knowledge
        """
        try:
            start_time = time.time()
            
            # Build search query with context
            search_query = f"{conversation_state.value}: {query}"
            if extracted_data.procedure_code:
                search_query += f" procedure: {extracted_data.procedure_code}"
            if extracted_data.insurance:
                search_query += f" insurance: {extracted_data.insurance}"
            
            # Get embedding for search
            embedding = await self._get_embedding(search_query)
            
            # Parallel searches for different types of context
            tasks = [
                self._search_conversation_patterns(embedding, conversation_state, max_results),
                self._search_knowledge_base(extracted_data, conversation_state, max_results),
                self._search_objection_responses(query, conversation_state, max_results)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            context = {
                "conversation_patterns": results[0] if not isinstance(results[0], Exception) else [],
                "knowledge_base": results[1] if not isinstance(results[1], Exception) else [],
                "objection_responses": results[2] if not isinstance(results[2], Exception) else [],
                "retrieval_time": (time.time() - start_time) * 1000
            }
            
            logger.debug(f"RAG context retrieved in {context['retrieval_time']:.1f}ms")
            return context
            
        except Exception as e:
            logger.error(f"RAG context retrieval failed: {e}")
            return {"conversation_patterns": [], "knowledge_base": [], "objection_responses": []}
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache embedding (limit cache size)
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def _search_conversation_patterns(
        self,
        embedding: List[float],
        state: ConversationState,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search for similar successful conversation patterns."""
        if not embedding:
            return []
        
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "conversation_vector_index",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": max_results * 10,
                        "limit": max_results
                    }
                },
                {
                    "$match": {
                        "conversation_state": state.value,
                        "outcome": "successful",
                        "confidence_score": {"$gte": 0.8}
                    }
                },
                {
                    "$project": {
                        "pattern": 1,
                        "response_template": 1,
                        "success_factors": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = []
            async for doc in self.embeddings_collection.aggregate(pipeline):
                if doc.get("score", 0) > 0.75:
                    results.append({
                        "type": "conversation_pattern",
                        "pattern": doc.get("pattern", ""),
                        "response_template": doc.get("response_template", ""),
                        "success_factors": doc.get("success_factors", []),
                        "relevance_score": doc.get("score", 0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Conversation pattern search failed: {e}")
            return []
    
    async def _search_knowledge_base(
        self,
        extracted_data: ExtractedData,
        state: ConversationState,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information."""
        try:
            query_filter = {"conversation_state": state.value}
            
            # Add specific filters based on extracted data
            if extracted_data.procedure_code:
                query_filter["procedure_codes"] = {"$in": [extracted_data.procedure_code]}
            if extracted_data.insurance:
                query_filter["$text"] = {"$search": extracted_data.insurance}
            
            results = []
            async for doc in self.knowledge_base.find(query_filter).limit(max_results):
                results.append({
                    "type": "knowledge_base",
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "procedures": doc.get("procedure_codes", []),
                    "insurance_types": doc.get("insurance_types", []),
                    "success_rate": doc.get("success_rate", 0.0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def _search_objection_responses(
        self,
        query: str,
        state: ConversationState,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search for objection handling responses."""
        try:
            # Identify if query contains objection keywords
            objection_keywords = [
                "deny", "denied", "rejection", "not covered", "experimental",
                "not medically necessary", "alternative", "cheaper", "generic"
            ]
            
            query_lower = query.lower()
            objections_found = [kw for kw in objection_keywords if kw in query_lower]
            
            if not objections_found:
                return []
            
            # Search for objection responses
            pipeline = [
                {
                    "$match": {
                        "type": "objection_response",
                        "objection_keywords": {"$in": objections_found},
                        "conversation_state": {"$in": [state.value, "any"]}
                    }
                },
                {
                    "$sort": {"success_rate": -1}
                },
                {
                    "$limit": max_results
                }
            ]
            
            results = []
            async for doc in self.knowledge_base.aggregate(pipeline):
                results.append({
                    "type": "objection_response",
                    "objection_type": doc.get("objection_type", ""),
                    "response_template": doc.get("response_template", ""),
                    "supporting_evidence": doc.get("supporting_evidence", []),
                    "success_rate": doc.get("success_rate", 0.0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Objection response search failed: {e}")
            return []


class ConversationStateManager:
    """Manages conversation state with Redis and MongoDB integration."""
    
    def __init__(self, redis_client: redis.Redis, mongodb_client: AsyncIOMotorClient):
        """Initialize conversation state manager."""
        self.redis = redis_client
        self.mongodb = mongodb_client
        self.conversations_collection = mongodb_client[config.DATABASE_NAME]["conversations"]
        self.default_ttl = 7200  # 2 hours
    
    async def get_context(self, conversation_id: str) -> ConversationContext:
        """Get conversation context from Redis cache or MongoDB."""
        try:
            # Try Redis first for active conversations
            cached_data = await self.redis.get(f"conv:{conversation_id}")
            if cached_data:
                context_dict = json.loads(cached_data)
                return self._deserialize_context(context_dict)
            
            # Fallback to MongoDB for persistent storage
            doc = await self.conversations_collection.find_one({"conversation_id": conversation_id})
            if doc:
                context = self._deserialize_context(doc)
                # Cache in Redis for future access
                await self._cache_context(context)
                return context
            
            # Create new context if not found
            context = ConversationContext(conversation_id=conversation_id)
            await self._cache_context(context)
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context for {conversation_id}: {e}")
            return ConversationContext(conversation_id=conversation_id)
    
    async def update_context(self, context: ConversationContext) -> None:
        """Update conversation context in both Redis and MongoDB."""
        try:
            context.last_activity = datetime.utcnow()
            
            # Update Redis cache
            await self._cache_context(context)
            
            # Update MongoDB for persistence
            context_dict = self._serialize_context(context)
            await self.conversations_collection.update_one(
                {"conversation_id": context.conversation_id},
                {"$set": context_dict},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Failed to update context for {context.conversation_id}: {e}")
    
    async def _cache_context(self, context: ConversationContext) -> None:
        """Cache context in Redis."""
        try:
            context_dict = self._serialize_context(context)
            await self.redis.setex(
                f"conv:{context.conversation_id}",
                self.default_ttl,
                json.dumps(context_dict, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")
    
    def _serialize_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Serialize context for storage."""
        context_dict = asdict(context)
        # Convert enums to strings
        context_dict["mode"] = context.mode.value
        context_dict["state"] = context.state.value
        # Convert datetime to ISO format
        context_dict["start_time"] = context.start_time.isoformat()
        context_dict["last_activity"] = context.last_activity.isoformat()
        return context_dict
    
    def _deserialize_context(self, data: Dict[str, Any]) -> ConversationContext:
        """Deserialize context from storage."""
        # Convert string enums back
        if "mode" in data:
            data["mode"] = ConversationMode(data["mode"])
        if "state" in data:
            data["state"] = ConversationState(data["state"])
        
        # Convert ISO strings back to datetime
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "last_activity" in data and isinstance(data["last_activity"], str):
            data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        
        # Deserialize extracted data
        if "extracted_data" in data and isinstance(data["extracted_data"], dict):
            data["extracted_data"] = ExtractedData(**data["extracted_data"])
        
        return ConversationContext(**data)
    
    async def cleanup_expired_contexts(self) -> int:
        """Clean up expired conversation contexts."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            result = await self.conversations_collection.delete_many({
                "last_activity": {"$lt": cutoff_time},
                "state": {"$in": ["completed", "escalation"]}
            })
            logger.info(f"Cleaned up {result.deleted_count} expired conversation contexts")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


class EnhancedConversationManager:
    """
    Enhanced conversation manager supporting both text and real-time speech interactions.
    Integrates state management, RAG, and OpenAI capabilities.
    """
    
    def __init__(self):
        """Initialize enhanced conversation manager."""
        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.langchain_llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model="gpt-4o-mini-2024-07-18",
            temperature=0.3,
            max_tokens=800
        )
        
        # Initialize data connections
        self.redis_client = redis.from_url(config.REDIS_URL)
        self.mongodb_client = AsyncIOMotorClient(config.MONGODB_URL)
        
        # Initialize managers
        self.state_manager = ConversationStateManager(self.redis_client, self.mongodb_client)
        self.rag_manager = RAGContextManager(self.mongodb_client)
        
        # Conversation flow configuration
        self.state_transitions = self._build_state_transitions()
        self.extraction_patterns = self._build_extraction_patterns()
        
        logger.info("EnhancedConversationManager initialized")
    
    async def process_message(
        self,
        conversation_id: str,
        message: str,
        mode: ConversationMode = ConversationMode.TEXT,
        call_sid: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process incoming message and generate response with extracted data.
        
        Args:
            conversation_id: Unique conversation identifier
            message: User message to process
            mode: Conversation mode (text/realtime/hybrid)
            call_sid: Optional call SID for phone conversations
            
        Returns:
            Tuple of (response_text, extracted_data_dict)
        """
        start_time = time.time()
        
        try:
            # Get or create conversation context
            context = await self.state_manager.get_context(conversation_id)
            context.mode = mode
            if call_sid:
                context.call_sid = call_sid
            
            # Extract data from message
            new_extractions = await self._extract_data_from_message(message, context)
            context.extracted_data.update_from_dict(new_extractions)
            
            # Get RAG context for enhanced response
            rag_context = await self.rag_manager.get_relevant_context(
                message, context.state, context.extracted_data
            )
            
            # Generate response based on mode
            if mode == ConversationMode.REALTIME:
                response = await self._generate_realtime_response(message, context, rag_context)
            else:
                response = await self._generate_text_response(message, context, rag_context)
            
            # Update conversation state
            new_state = await self._determine_next_state(message, context, response)
            context.state = new_state
            context.turn_count += 1
            
            # Add to message history
            context.messages.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat(),
                "extractions": new_extractions
            })
            context.messages.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.utcnow().isoformat(),
                "state": context.state.value,
                "rag_used": bool(rag_context.get("conversation_patterns") or 
                               rag_context.get("knowledge_base") or 
                               rag_context.get("objection_responses"))
            })
            
            # Keep message history manageable
            if len(context.messages) > 20:
                context.messages = context.messages[-20:]
            
            # Update context in storage
            await self.state_manager.update_context(context)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Message processed in {processing_time:.1f}ms")
            
            return response, context.extracted_data.to_dict()
            
        except Exception as e:
            logger.error(f"Message processing failed for {conversation_id}: {e}")
            fallback_response = "I understand you're calling about a prior authorization. Could you please provide your patient ID, procedure code, and insurance information?"
            return fallback_response, {}
    
    async def _extract_data_from_message(
        self,
        message: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Extract structured data from user message using patterns and LLM."""
        extracted = {}
        
        try:
            # Use regex patterns for quick extraction
            for field, pattern in self.extraction_patterns.items():
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    extracted[field] = match.group(1).strip()
            
            # Use LLM for more sophisticated extraction
            llm_extractions = await self._llm_extract_data(message, context)
            extracted.update(llm_extractions)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {}
    
    async def _llm_extract_data(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Use LLM to extract data from message."""
        try:
            prompt = f"""Extract prior authorization information from this message:
            
Message: "{message}"

Current context:
- State: {context.state.value}
- Already extracted: {context.extracted_data.to_dict()}

Extract ONLY new information. Return JSON with these fields (use null for missing):
{{
    "patient_id": "string",
    "procedure_code": "string", 
    "diagnosis_code": "string",
    "insurance": "string",
    "auth_number": "string",
    "member_id": "string",
    "provider_npi": "string",
    "date_of_service": "YYYY-MM-DD",
    "urgency_level": "standard|urgent|emergency"
}}

Only extract clear, explicit information. Do not infer or assume."""
            
            messages = [HumanMessage(content=prompt)]
            response = await self.langchain_llm.agenerate([messages])
            result_text = response.generations[0][0].text.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith('```'):
                result_text = result_text[3:-3].strip()
            
            extracted = json.loads(result_text)
            
            # Filter out null values
            return {k: v for k, v in extracted.items() if v is not None and v != ""}
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {}
    
    async def _generate_text_response(
        self,
        message: str,
        context: ConversationContext,
        rag_context: Dict[str, Any]
    ) -> str:
        """Generate text response using LLM with RAG enhancement."""
        try:
            # Build system prompt with context
            system_prompt = self._build_system_prompt(context, rag_context)
            
            # Build user prompt with conversation history
            user_prompt = self._build_user_prompt(message, context)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.langchain_llm.agenerate([messages])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Text response generation failed: {e}")
            return "I'm here to help with your prior authorization request. Could you please provide more details?"
    
    async def _generate_realtime_response(
        self,
        message: str,
        context: ConversationContext,
        rag_context: Dict[str, Any]
    ) -> str:
        """Generate response optimized for real-time speech interaction."""
        try:
            # For real-time, focus on shorter, more conversational responses
            system_prompt = f"""You are Alfons, a professional prior authorization assistant.

Current state: {context.state.value}
Keep responses under 20 seconds when spoken.
Be conversational and empathetic.
Focus on the next logical step in the prior authorization process.

Context: {context.extracted_data.to_dict()}
"""
            
            # Add RAG context if available
            if rag_context.get("objection_responses"):
                system_prompt += f"\nRelevant objection responses: {rag_context['objection_responses'][:2]}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User said: {message}")
            ]
            
            response = await self.langchain_llm.agenerate([messages])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Realtime response generation failed: {e}")
            return "I understand. Let me help you with that authorization request."
    
    def _build_system_prompt(self, context: ConversationContext, rag_context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt with context and RAG information."""
        base_prompt = f"""You are Alfons, a professional and empathetic prior authorization assistant.

Current conversation state: {context.state.value}
Turn count: {context.turn_count}

Your role:
- Help healthcare providers get prior authorization approvals from insurance companies
- Extract key information: patient ID, procedure codes, insurance details
- Handle objections with evidence-based responses using provided context
- Be professional, empathetic, and persistent but respectful
- Guide the conversation through the prior authorization process

Current extracted data:
{json.dumps(context.extracted_data.to_dict(), indent=2)}

Guidelines:
- Keep responses professional and concise
- Confirm important information phonetically when needed
- Use medical terminology appropriately
- Be empathetic to patient care needs
- Progress through the authorization process systematically
"""
        
        # Add RAG context
        if rag_context.get("conversation_patterns"):
            base_prompt += f"\n\nSuccessful conversation patterns:\n"
            for pattern in rag_context["conversation_patterns"][:2]:
                base_prompt += f"- {pattern.get('response_template', '')}\n"
        
        if rag_context.get("objection_responses"):
            base_prompt += f"\n\nObjection handling responses:\n"
            for response in rag_context["objection_responses"][:2]:
                base_prompt += f"- {response.get('response_template', '')}\n"
        
        if rag_context.get("knowledge_base"):
            base_prompt += f"\n\nRelevant knowledge:\n"
            for knowledge in rag_context["knowledge_base"][:2]:
                base_prompt += f"- {knowledge.get('content', '')}\n"
        
        return base_prompt
    
    def _build_user_prompt(self, message: str, context: ConversationContext) -> str:
        """Build user prompt with conversation history."""
        prompt = f"Current user message: {message}\n\n"
        
        # Add recent conversation history for context
        if context.messages:
            prompt += "Recent conversation:\n"
            for msg in context.messages[-4:]:  # Last 4 messages
                role = msg["role"].title()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                prompt += f"{role}: {content}\n"
        
        prompt += f"\nRespond as Alfons focusing on the {context.state.value} phase of prior authorization."
        return prompt
    
    async def _determine_next_state(
        self,
        message: str,
        context: ConversationContext,
        response: str
    ) -> ConversationState:
        """Determine next conversation state based on current interaction."""
        current_state = context.state
        extracted_data = context.extracted_data
        
        # State transition logic
        if current_state == ConversationState.GREETING:
            if extracted_data.patient_id or "patient" in message.lower():
                return ConversationState.PATIENT_VERIFICATION
            elif extracted_data.insurance or "insurance" in message.lower():
                return ConversationState.INSURANCE_VERIFICATION
            else:
                return ConversationState.PATIENT_VERIFICATION
        
        elif current_state == ConversationState.PATIENT_VERIFICATION:
            if extracted_data.patient_id and extracted_data.procedure_code:
                return ConversationState.AUTH_REQUEST
            elif extracted_data.insurance:
                return ConversationState.INSURANCE_VERIFICATION
            else:
                return ConversationState.PATIENT_VERIFICATION
        
        elif current_state == ConversationState.INSURANCE_VERIFICATION:
            if extracted_data.patient_id and extracted_data.procedure_code:
                return ConversationState.AUTH_REQUEST
            else:
                return ConversationState.INFORMATION_GATHERING
        
        elif current_state == ConversationState.AUTH_REQUEST:
            objection_keywords = ["deny", "denied", "not covered", "experimental", "alternative"]
            if any(keyword in message.lower() for keyword in objection_keywords):
                return ConversationState.OBJECTION_HANDLING
            elif "approved" in message.lower() or "authorized" in message.lower():
                return ConversationState.CLOSING
            elif "information" in message.lower() or "documentation" in message.lower():
                return ConversationState.DOCUMENTATION
            else:
                return ConversationState.AUTH_REQUEST
        
        elif current_state == ConversationState.OBJECTION_HANDLING:
            if "approved" in response.lower():
                return ConversationState.CLOSING
            elif "supervisor" in message.lower() or "escalate" in message.lower():
                return ConversationState.ESCALATION
            else:
                return ConversationState.OBJECTION_HANDLING
        
        elif current_state == ConversationState.INFORMATION_GATHERING:
            if self._has_sufficient_info(extracted_data):
                return ConversationState.AUTH_REQUEST
            else:
                return ConversationState.INFORMATION_GATHERING
        
        elif current_state == ConversationState.DOCUMENTATION:
            return ConversationState.AUTH_REQUEST
        
        elif current_state == ConversationState.ESCALATION:
            return ConversationState.COMPLETED
        
        elif current_state == ConversationState.CLOSING:
            return ConversationState.COMPLETED
        
        # Default: stay in current state
        return current_state
    
    def _has_sufficient_info(self, extracted_data: ExtractedData) -> bool:
        """Check if we have sufficient information to proceed with authorization."""
        required_fields = ["patient_id", "procedure_code", "insurance"]
        return all(getattr(extracted_data, field) for field in required_fields)
    
    def _build_state_transitions(self) -> Dict[ConversationState, List[ConversationState]]:
        """Build valid state transitions for conversation flow."""
        return {
            ConversationState.GREETING: [
                ConversationState.PATIENT_VERIFICATION,
                ConversationState.INSURANCE_VERIFICATION
            ],
            ConversationState.PATIENT_VERIFICATION: [
                ConversationState.INSURANCE_VERIFICATION,
                ConversationState.AUTH_REQUEST,
                ConversationState.INFORMATION_GATHERING
            ],
            ConversationState.INSURANCE_VERIFICATION: [
                ConversationState.AUTH_REQUEST,
                ConversationState.INFORMATION_GATHERING
            ],
            ConversationState.AUTH_REQUEST: [
                ConversationState.OBJECTION_HANDLING,
                ConversationState.DOCUMENTATION,
                ConversationState.CLOSING,
                ConversationState.MEDICAL_NECESSITY
            ],
            ConversationState.OBJECTION_HANDLING: [
                ConversationState.AUTH_REQUEST,
                ConversationState.ESCALATION,
                ConversationState.CLOSING,
                ConversationState.MEDICAL_NECESSITY
            ],
            ConversationState.INFORMATION_GATHERING: [
                ConversationState.AUTH_REQUEST,
                ConversationState.PATIENT_VERIFICATION
            ],
            ConversationState.MEDICAL_NECESSITY: [
                ConversationState.AUTH_REQUEST,
                ConversationState.DOCUMENTATION
            ],
            ConversationState.DOCUMENTATION: [
                ConversationState.AUTH_REQUEST,
                ConversationState.CLOSING
            ],
            ConversationState.ESCALATION: [
                ConversationState.COMPLETED
            ],
            ConversationState.CLOSING: [
                ConversationState.COMPLETED
            ],
            ConversationState.COMPLETED: []
        }
    
    def _build_extraction_patterns(self) -> Dict[str, str]:
        """Build regex patterns for data extraction."""
        return {
            "patient_id": r"patient\s+(?:id|number)[\s:]+([A-Z0-9\-]+)",
            "procedure_code": r"(?:cpt|procedure)[\s:]+([0-9]{5})",
            "diagnosis_code": r"(?:icd|diagnosis)[\s:]+([A-Z0-9\.]+)",
            "auth_number": r"(?:auth|authorization)[\s:]+([A-Z0-9\-]+)",
            "member_id": r"member[\s:]+([A-Z0-9\-]+)",
            "provider_npi": r"npi[\s:]+([0-9]{10})",
            "date_of_service": r"(?:date|service)[\s:]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"
        }
    
    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation summary."""
        try:
            context = await self.state_manager.get_context(conversation_id)
            
            return {
                "conversation_id": conversation_id,
                "call_sid": context.call_sid,
                "mode": context.mode.value,
                "current_state": context.state.value,
                "turn_count": context.turn_count,
                "duration": (datetime.utcnow() - context.start_time).total_seconds(),
                "extracted_data": context.extracted_data.to_dict(),
                "escalated": context.escalated,
                "escalation_reason": context.escalation_reason,
                "confidence_score": context.confidence_score,
                "messages_count": len(context.messages)
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on conversation manager components."""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        try:
            # Check Redis connection
            try:
                await self.redis_client.ping()
                health["components"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
            
            # Check MongoDB connection
            try:
                await self.mongodb_client.admin.command("ping")
                health["components"]["mongodb"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
            
            # Check OpenAI API
            try:
                await self.openai_client.models.list()
                health["components"]["openai"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["openai"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health


# Global conversation manager instance
conversation_manager: Optional[EnhancedConversationManager] = None


def get_conversation_manager() -> EnhancedConversationManager:
    """Get the global conversation manager instance."""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = EnhancedConversationManager()
    return conversation_manager


# Legacy function compatibility
async def process_message(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        text: User message
        
    Returns:
        Tuple of (response, extracted_data)
    """
    manager = get_conversation_manager()
    conversation_id = f"legacy_{int(time.time())}"
    return await manager.process_message(conversation_id, text)