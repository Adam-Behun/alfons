# Abstract interface for LLM providers
# Enables swapping between OpenAI, Anthropic, XAI, etc. via configuration

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
import asyncio

class LLMMessage(BaseModel):
    """Standardized message format across providers"""
    role: str  # "system", "user", "assistant"
    content: str
    
class LLMResponse(BaseModel):
    """Standardized response format across providers"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    latency_ms: Optional[float] = None

class StreamingLLMResponse(BaseModel):
    """Standardized streaming response chunk"""
    content: str
    is_final: bool = False
    usage: Optional[Dict[str, Any]] = None

class ILLMProvider(ABC):
    """
    Abstract interface for Large Language Model providers.
    Ensures consistent API across OpenAI, Anthropic, XAI, etc.
    """
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 500,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a single response from the LLM
        
        Args:
            messages: List of conversation messages
            temperature: Randomness (0.0 - 1.0)
            max_tokens: Maximum response tokens
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with generated content and metadata
        """
        pass
    
    @abstractmethod
    async def stream_response(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 500,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """
        Stream response tokens as they're generated
        
        Args:
            messages: List of conversation messages
            temperature: Randomness (0.0 - 1.0)
            max_tokens: Maximum response tokens
            **kwargs: Provider-specific parameters
            
        Yields:
            StreamingLLMResponse chunks
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Test if the provider connection is working
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        pass
    
    @property 
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Whether this provider supports function calling"""
        pass
    
    def format_healthcare_prompt(self, base_prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Format prompt with healthcare-specific context and instructions
        
        Args:
            base_prompt: Base prompt template
            context: Additional context (patient_id, procedure_code, etc.)
            
        Returns:
            Formatted prompt with healthcare context
        """
        healthcare_context = """
# Healthcare AI Assistant Context
You are Alfons, a professional healthcare prior authorization specialist.
- Always maintain HIPAA compliance and patient confidentiality
- Use precise medical terminology and CPT codes
- Confirm critical information using phonetic spelling
- Keep responses concise for phone conversations (<25 words typically)
- Escalate complex cases to human representatives when appropriate
"""
        
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items() if v])
            healthcare_context += f"\n# Current Context:\n{context_str}\n"
        
        return f"{healthcare_context}\n\n{base_prompt}"

from langchain_openai import ChatOpenAI
from shared.config import config

def get_llm(model: str = "gpt-4", **kwargs):
    return ChatOpenAI(api_key=config.OPENAI_API_KEY, model=model, **kwargs)