"""
Core Speech-to-Speech pipeline for real-time healthcare prior authorization calls.
Implements <500ms latency using OpenAI Realtime API with RAG integration.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io
import wave

from openai import AsyncOpenAI
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


class CallState(Enum):
    """Call state machine for prior authorization flow."""
    GREETING = "greeting"
    PATIENT_VERIFICATION = "patient_verification"
    AUTH_REQUEST = "auth_request"
    OBJECTION_HANDLING = "objection_handling"
    INFORMATION_GATHERING = "information_gathering"
    CLOSING = "closing"
    ESCALATION = "escalation"


@dataclass
class ConversationMemory:
    """Working memory for current conversation."""
    call_id: str
    patient_id: Optional[str] = None
    procedure_code: Optional[str] = None
    insurance: Optional[str] = None
    auth_number: Optional[str] = None
    current_state: CallState = CallState.GREETING
    turn_count: int = 0
    context_history: List[str] = None
    extracted_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_history is None:
            self.context_history = []
        if self.extracted_data is None:
            self.extracted_data = {}


@dataclass
class S2SConfig:
    """Configuration for S2S pipeline."""
    # OpenAI Realtime
    model: str = "gpt-4o-realtime-preview-2025-06-03"
    voice: str = "alloy"  # Professional, empathetic voice for healthcare
    
    # Latency targets (ms)
    max_response_time: int = 500
    stream_buffer_ms: int = 200
    
    # RAG settings
    enable_rag: bool = True
    max_rag_results: int = 3
    rag_timeout_ms: int = 50
    
    # Memory settings
    redis_ttl: int = 3600  # 1 hour working memory
    context_window: int = 10  # Last 10 turns
    
    # Healthcare settings
    enable_hipaa_filter: bool = True
    require_patient_verification: bool = True
    escalation_keywords: List[str] = None
    
    def __post_init__(self):
        if self.escalation_keywords is None:
            self.escalation_keywords = [
                "supervisor", "manager", "escalate", "complaint", 
                "legal", "lawyer", "violation", "urgent"
            ]


class RAGRetriever:
    """Retrieves relevant context from past prior auth conversations."""
    
    def __init__(self, mongodb_client: AsyncIOMotorClient):
        """Initialize RAG retriever with MongoDB client and proper config."""
        self.mongodb_client = mongodb_client
        
        # Use the database name from the global config
        self.db = mongodb_client[config.DATABASE_NAME]
        self.embeddings_collection = self.db["conversation_embeddings"]
        
        # Initialize OpenAI client with config API key
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured in settings")
        
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        logger.info(f"RAGRetriever initialized with database: {config.DATABASE_NAME}")
    
    async def get_relevant_context(
        self, 
        query: str, 
        call_state: CallState,
        max_results: int = 3
    ) -> List[str]:
        """
        Retrieve relevant context from past conversations.
        
        Args:
            query: Current conversation context
            call_state: Current state in conversation flow
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant context strings
        """
        try:
            start_time = time.time()
            
            # Generate embedding for query
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=f"{call_state.value}: {query}"
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Vector search in MongoDB
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "conversation_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": max_results * 10,
                        "limit": max_results
                    }
                },
                {
                    "$match": {
                        "call_state": call_state.value,
                        "success_outcome": True  # Only retrieve successful patterns
                    }
                },
                {
                    "$project": {
                        "context": 1,
                        "response_pattern": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = []
            async for doc in self.embeddings_collection.aggregate(pipeline):
                if doc.get("score", 0) > 0.7:  # Similarity threshold
                    context = f"Pattern: {doc.get('response_pattern', '')}"
                    results.append(context)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"RAG retrieval completed in {processing_time:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []


class ConversationStateManager:
    """Manages conversation state using Redis for low-latency access."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize state manager with Redis client."""
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
        
        logger.info("ConversationStateManager initialized")
    
    async def get_memory(self, call_id: str) -> ConversationMemory:
        """Get conversation memory from Redis."""
        try:
            data = await self.redis.get(f"call:{call_id}")
            if data:
                memory_dict = json.loads(data)
                # Convert state string back to enum
                if "current_state" in memory_dict:
                    memory_dict["current_state"] = CallState(memory_dict["current_state"])
                return ConversationMemory(**memory_dict)
            else:
                # Initialize new conversation
                return ConversationMemory(call_id=call_id)
        except Exception as e:
            logger.error(f"Failed to get memory for {call_id}: {e}")
            return ConversationMemory(call_id=call_id)
    
    async def update_memory(self, memory: ConversationMemory) -> None:
        """Update conversation memory in Redis."""
        try:
            # Convert to dict and handle enum serialization
            memory_dict = asdict(memory)
            memory_dict["current_state"] = memory.current_state.value
            
            await self.redis.setex(
                f"call:{memory.call_id}",
                self.default_ttl,
                json.dumps(memory_dict)
            )
        except Exception as e:
            logger.error(f"Failed to update memory for {memory.call_id}: {e}")
    
    async def cleanup_memory(self, call_id: str) -> None:
        """Clean up conversation memory after call ends."""
        try:
            await self.redis.delete(f"call:{call_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup memory for {call_id}: {e}")


class S2SPipeline:
    """
    Core Speech-to-Speech pipeline for healthcare prior authorization calls.
    Integrates OpenAI Realtime API with RAG and state management.
    """
    
    def __init__(self, s2s_config: S2SConfig = None):
        """Initialize S2S pipeline with components."""
        self.s2s_config = s2s_config or S2SConfig()
        
        # Validate required configuration
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured in settings")
        if not config.MONGODB_URL:
            raise ValueError("MONGODB_URL not configured in settings")
        if not config.REDIS_URL:
            raise ValueError("REDIS_URL not configured in settings")
        
        # Initialize clients using global config
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.redis_client = redis.from_url(config.REDIS_URL)
        self.mongodb_client = AsyncIOMotorClient(config.MONGODB_URL)
        
        # Initialize components
        self.rag_retriever = RAGRetriever(self.mongodb_client)
        self.state_manager = ConversationStateManager(self.redis_client)
        
        # Active realtime sessions
        self.active_sessions: Dict[str, Any] = {}
        
        logger.info("S2SPipeline initialized for healthcare prior auth")
        logger.info(f"Using database: {config.DATABASE_NAME}")
        logger.info(f"Using model: {self.s2s_config.model}")
    
    async def start_call_session(self, call_id: str) -> Dict[str, Any]:
        """
        Start a new S2S session for a call.
        
        Args:
            call_id: Unique identifier for the call
            
        Returns:
            Session initialization data
        """
        logger.info(f"Starting S2S session for call: {call_id}")
        
        try:
            # Initialize conversation memory
            memory = await self.state_manager.get_memory(call_id)
            
            # Create Realtime session configuration
            session_config = {
                "model": self.s2s_config.model,
                "voice": self.s2s_config.voice,
                "instructions": self._build_system_instructions(memory),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "tools": self._build_tools_config(),
                "temperature": 0.7,
                "max_response_output_tokens": 150  # Keep responses concise
            }
            
            # Store session info
            self.active_sessions[call_id] = {
                "config": session_config,
                "memory": memory,
                "start_time": time.time(),
                "latency_stats": {"response_times": []}
            }
            
            return {
                "status": "initialized",
                "call_id": call_id,
                "session_config": session_config
            }
            
        except Exception as e:
            logger.error(f"Failed to start session for {call_id}: {e}")
            raise
    
    async def process_audio_chunk(
        self, 
        call_id: str, 
        audio_data: bytes,
        session_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Process incoming audio chunk and yield response audio.
        
        Args:
            call_id: Call identifier
            audio_data: Incoming audio data (μ-law from Twilio)
            session_context: Additional session context
            
        Yields:
            Response audio chunks
        """
        if call_id not in self.active_sessions:
            logger.error(f"No active session for call: {call_id}")
            return
        
        session = self.active_sessions[call_id]
        start_time = time.time()
        
        try:
            # Convert μ-law to PCM16 for OpenAI
            pcm_audio = self._convert_mulaw_to_pcm16(audio_data)
            
            # Get current memory
            memory = await self.state_manager.get_memory(call_id)
            
            # Create realtime session if not exists
            if "realtime_session" not in session:
                session["realtime_session"] = await self._create_realtime_session(
                    session["config"], memory
                )
            
            realtime_session = session["realtime_session"]
            
            # Send audio to realtime API
            await realtime_session.input_audio_buffer.append(pcm_audio)
            
            # Process with RAG enhancement if enabled
            if self.s2s_config.enable_rag:
                # Get partial transcription for context
                transcript = await self._get_partial_transcript(realtime_session)
                if transcript:
                    rag_context = await self.rag_retriever.get_relevant_context(
                        transcript, memory.current_state, self.s2s_config.max_rag_results
                    )
                    if rag_context:
                        await self._inject_rag_context(realtime_session, rag_context)
            
            # Generate and stream response
            async for response_chunk in realtime_session.generate_response():
                # Convert PCM16 back to μ-law for Twilio
                mulaw_chunk = self._convert_pcm16_to_mulaw(response_chunk.audio)
                yield mulaw_chunk
                
                # Update conversation state
                if response_chunk.is_final:
                    await self._update_conversation_state(
                        memory, response_chunk.transcript, response_chunk.audio_transcript
                    )
            
            # Track latency
            response_time = (time.time() - start_time) * 1000
            session["latency_stats"]["response_times"].append(response_time)
            
            if response_time > self.s2s_config.max_response_time:
                logger.warning(f"Response time {response_time:.1f}ms exceeded target {self.s2s_config.max_response_time}ms")
            
        except Exception as e:
            logger.error(f"Error processing audio for {call_id}: {e}")
    
    async def end_call_session(self, call_id: str) -> Dict[str, Any]:
        """
        End S2S session and cleanup resources.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Session summary and metrics
        """
        logger.info(f"Ending S2S session for call: {call_id}")
        
        try:
            if call_id not in self.active_sessions:
                logger.warning(f"No active session found for {call_id}")
                return {"status": "no_session"}
            
            session = self.active_sessions[call_id]
            
            # Calculate session metrics
            total_time = time.time() - session["start_time"]
            avg_response_time = sum(session["latency_stats"]["response_times"]) / len(session["latency_stats"]["response_times"]) if session["latency_stats"]["response_times"] else 0
            
            # Get final conversation state
            memory = await self.state_manager.get_memory(call_id)
            
            # Close realtime session
            if "realtime_session" in session:
                await session["realtime_session"].close()
            
            # Cleanup
            del self.active_sessions[call_id]
            await self.state_manager.cleanup_memory(call_id)
            
            summary = {
                "status": "completed",
                "call_id": call_id,
                "total_duration": total_time,
                "average_response_time": avg_response_time,
                "turn_count": memory.turn_count,
                "final_state": memory.current_state.value,
                "extracted_data": memory.extracted_data
            }
            
            logger.info(f"Session completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error ending session for {call_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _build_system_instructions(self, memory: ConversationMemory) -> str:
        """Build system instructions for OpenAI Realtime."""
        base_instructions = f"""You are Alfons, a professional and empathetic prior authorization assistant for healthcare providers.

Current conversation state: {memory.current_state.value}
Turn count: {memory.turn_count}

Your role:
- Help healthcare providers get prior authorization approvals from insurance companies
- Be professional, empathetic, and persistent but respectful
- Extract key information: patient ID, procedure codes, insurance details
- Handle objections with evidence-based responses
- Escalate complex cases when necessary

Guidelines:
- Keep responses concise (under 30 seconds)
- Use professional medical terminology
- Confirm important information phonetically
- Be empathetic to patient care needs
- Never share or request protected health information inappropriately

Current conversation context:
Patient ID: {memory.patient_id or 'Not provided'}
Procedure: {memory.procedure_code or 'Not specified'}
Insurance: {memory.insurance or 'Unknown'}
"""
        
        # Add state-specific instructions
        state_instructions = {
            CallState.GREETING: "Start with a professional greeting and identify the purpose of the call.",
            CallState.PATIENT_VERIFICATION: "Verify patient information and gather necessary details.",
            CallState.AUTH_REQUEST: "Present the authorization request clearly with supporting information.",
            CallState.OBJECTION_HANDLING: "Address objections with empathy and evidence. Use RAG context if available.",
            CallState.INFORMATION_GATHERING: "Collect any additional information needed for approval.",
            CallState.CLOSING: "Summarize the outcome and next steps clearly.",
            CallState.ESCALATION: "Professional escalation while maintaining relationship."
        }
        
        if memory.current_state in state_instructions:
            base_instructions += f"\n\nCurrent focus: {state_instructions[memory.current_state]}"
        
        return base_instructions
    
    def _build_tools_config(self) -> List[Dict[str, Any]]:
        """Build tools configuration for function calling."""
        return [
            {
                "name": "extract_patient_info",
                "description": "Extract and store patient information from conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "procedure_code": {"type": "string"},
                        "insurance": {"type": "string"},
                        "auth_number": {"type": "string"}
                    }
                }
            },
            {
                "name": "update_call_state",
                "description": "Update the current state of the prior authorization call",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_state": {
                            "type": "string",
                            "enum": [state.value for state in CallState]
                        },
                        "reason": {"type": "string"}
                    }
                }
            },
            {
                "name": "request_escalation",
                "description": "Request escalation to human supervisor",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "urgency": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                }
            }
        ]
    
    def _convert_mulaw_to_pcm16(self, mulaw_data: bytes) -> bytes:
        """Convert μ-law audio from Twilio to PCM16 for OpenAI."""
        # Implementation would use audioop or similar for μ-law conversion
        # For now, placeholder that assumes conversion is handled
        return mulaw_data  # Placeholder - implement actual conversion
    
    def _convert_pcm16_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convert PCM16 audio to μ-law for Twilio."""
        # Implementation would use audioop or similar for μ-law conversion
        return pcm_data  # Placeholder - implement actual conversion
    
    async def _create_realtime_session(self, config: Dict[str, Any], memory: ConversationMemory):
        """Create OpenAI Realtime session (placeholder for actual implementation)."""
        # This would create the actual OpenAI Realtime session
        # Implementation depends on final OpenAI Realtime Python SDK
        pass
    
    async def _get_partial_transcript(self, session) -> Optional[str]:
        """Get partial transcript for RAG context."""
        # Implementation would extract partial transcript from realtime session
        return None  # Placeholder
    
    async def _inject_rag_context(self, session, rag_context: List[str]):
        """Inject RAG context into the realtime session."""
        # Implementation would add context to the session
        pass
    
    async def _update_conversation_state(self, memory: ConversationMemory, user_transcript: str, assistant_transcript: str):
        """Update conversation state based on latest exchange."""
        memory.turn_count += 1
        memory.context_history.append(f"User: {user_transcript}")
        memory.context_history.append(f"Assistant: {assistant_transcript}")
        
        # Keep only recent context
        if len(memory.context_history) > self.s2s_config.context_window * 2:
            memory.context_history = memory.context_history[-self.s2s_config.context_window * 2:]
        
        # Update state in Redis
        await self.state_manager.update_memory(memory)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all S2S components."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # Check OpenAI client
            health_status["components"]["openai"] = {
                "status": "healthy" if config.OPENAI_API_KEY else "missing_key"
            }
            
            # Check Redis connection
            try:
                await self.redis_client.ping()
                health_status["components"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
                health_status["status"] = "degraded"
            
            # Check MongoDB connection
            try:
                await self.mongodb_client.admin.command('ping')
                health_status["components"]["mongodb"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
                health_status["status"] = "degraded"
            
            # Check active sessions
            health_status["components"]["sessions"] = {
                "status": "healthy",
                "active_count": len(self.active_sessions)
            }
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


# Factory function for easy initialization
def create_s2s_pipeline(
    model: str = "gpt-4o-realtime-preview-2025-06-03",
    voice: str = "alloy",
    enable_rag: bool = True
) -> S2SPipeline:
    """
    Create S2S pipeline with healthcare-optimized configuration.
    
    Args:
        model: OpenAI Realtime model to use
        voice: Voice for TTS output
        enable_rag: Enable RAG for enhanced responses
        
    Returns:
        Configured S2SPipeline instance
    """
    s2s_config = S2SConfig(
        model=model,
        voice=voice,
        enable_rag=enable_rag,
        enable_hipaa_filter=True,
        require_patient_verification=True
    )
    
    return S2SPipeline(s2s_config)