import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import redis.asyncio as redis

from shared.logging import get_logger
from shared.providers.illm_provider import get_llm
from call_analytics.mongo_connector import AsyncMongoConnector

logger = get_logger(__name__)

class ConversationState(Enum):
    """Simplified prior auth states."""
    GREETING = "greeting"
    VERIFICATION = "verification"
    AUTH_REQUEST = "auth_request"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    COMPLETED = "completed"

@dataclass
class ExtractedData:
    """Extracted data from conversation."""
    patient_id: Optional[str] = None
    procedure_code: Optional[str] = None
    insurance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key) and value:
                setattr(self, key, value)

@dataclass
class ConversationContext:
    """Conversation context/state."""
    conversation_id: str
    state: ConversationState = ConversationState.GREETING
    extracted_data: ExtractedData = field(default_factory=ExtractedData)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)

class EnhancedConversationManager:
    """
    Simplified conversation manager: state tracking, LLM responses, data extraction.
    Uses LLM provider; in-memory for MVP.
    """

    def __init__(self):
        self.llm = get_llm(model="gpt-4", temperature=0.3, max_tokens=800)
        self.parser = JsonOutputParser()
        self.conversations: Dict[str, ConversationContext] = {}
        logger.info("EnhancedConversationManager initialized")

    async def process_message(
        self,
        conversation_id: str,
        message: str,
        call_sid: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process message: extract data, generate response, update state.

        :param conversation_id: ID.
        :param message: User message.
        :param call_sid: Optional SID.
        :return: Response, extracted data.
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id)

        context = self.conversations[conversation_id]
        context.messages.append({"role": "user", "content": message})

        redis_client = await redis.from_url(config.REDIS_URL)
        channel = f"transcript_{call_sid or conversation_id}"

        # CoT Prompt Chain
        cot_prompt = ChatPromptTemplate.from_template(
            """Think step by step about the prior auth task in state {state}:
Extracted data: {extracted}
User message: {message}

Output JSON:
{{
  "thoughts": "Step 1: ... Step 2: ... (detailed reasoning)",
  "response": "Professional, concise response"
}}"""
        )
        chain = cot_prompt | self.llm | self.parser  # LCEL for streaming

        full_thoughts = ""
        full_response = ""
        in_thoughts = True  # Track parsing state

        async for chunk in chain.astream({"state": context.state.value, "extracted": context.extracted_data.to_dict(), "message": message}):
            # Chunk is partial dict, e.g., { "thoughts": "Step 1: " } then additions
            if isinstance(chunk, dict):
                thoughts_chunk = chunk.get("thoughts", "")
                response_chunk = chunk.get("response", "")
                if thoughts_chunk:
                    full_thoughts += thoughts_chunk
                    await redis_client.publish(channel, json.dumps({"type": "chunk", "role": "assistant_thought", "chunk": thoughts_chunk}))
                if response_chunk:
                    in_thoughts = False
                    full_response += response_chunk
                    await redis_client.publish(channel, json.dumps({"type": "chunk", "role": "assistant", "chunk": response_chunk}))

        # On complete: extract, publish, store
        extracted = await self._extract_data(full_response, context)
        context.extracted_data.update_from_dict(extracted)
        await redis_client.publish(channel, json.dumps({
            "type": "complete", "role": "assistant", "thoughts": full_thoughts, "content": full_response,
            "extracted": extracted, "timestamp": datetime.utcnow().isoformat()
        }))

        mongo = AsyncMongoConnector()
        await mongo.connect()
        await mongo.insert_document("logs", {
            "call_sid": call_sid,
            "user_input": message,
            "bot_thoughts": full_thoughts,  # New field for auditing/learning
            "bot_response": full_response,
            **extracted,
            "timestamp": datetime.utcnow().isoformat()
        })
        await mongo.close_connection()

        await redis_client.close()
        context.messages.append({"role": "assistant", "content": full_response})
        context.state = self._determine_next_state(message, context, full_response)
        return full_response, context.extracted_data.to_dict()

    async def _extract_data(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Extract via LLM.

        :param message: Message.
        :param context: Context.
        :return: Extractions dict.
        """
        prompt = f"""Extract prior auth info from: "{message}"
Current: {context.extracted_data.to_dict()}
Return JSON: {{"patient_id": "str or null", "procedure_code": "str or null", "insurance": "str or null"}}
Only new explicit info."""
        try:
            response = await self.llm.invoke([HumanMessage(content=prompt)])
            extracted = json.loads(response.content.strip())
            return {k: v for k, v in extracted.items() if v}
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {}

    def _determine_next_state(self, message: str, context: ConversationContext, response: str) -> ConversationState:
        """
        Next state logic.

        :param message: Message.
        :param context: Context.
        :param response: Response.
        :return: New state.
        """
        current = context.state
        data = context.extracted_data

        if current == ConversationState.GREETING:
            return ConversationState.VERIFICATION

        if current == ConversationState.VERIFICATION:
            if data.patient_id and data.procedure_code and data.insurance:
                return ConversationState.AUTH_REQUEST
            return ConversationState.VERIFICATION

        if current == ConversationState.AUTH_REQUEST:
            if "deny" in message.lower() or "not covered" in message.lower():
                return ConversationState.OBJECTION_HANDLING
            if "approved" in response.lower():
                return ConversationState.CLOSING
            return ConversationState.AUTH_REQUEST

        if current == ConversationState.OBJECTION_HANDLING:
            if "approved" in response.lower():
                return ConversationState.CLOSING
            return ConversationState.OBJECTION_HANDLING

        if current == ConversationState.CLOSING:
            return ConversationState.COMPLETED

        return current

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get summary.

        :param conversation_id: ID.
        :return: Summary dict.
        """
        if conversation_id not in self.conversations:
            return {"error": "Not found"}

        context = self.conversations[conversation_id]
        duration = (datetime.utcnow() - context.start_time).total_seconds()
        return {
            "id": conversation_id,
            "state": context.state.value,
            "messages": len(context.messages),
            "duration": duration,
            "extracted": context.extracted_data.to_dict()
        }

    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "active": len(self.conversations)}

def get_conversation_manager() -> EnhancedConversationManager:
    return EnhancedConversationManager()