import logging
from typing import Dict, Any, Optional, List
import asyncio
from langchain_openai import ChatOpenAI

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class ScriptGenerator:
    """
    Generates scripts/responses using context-aware OpenAI prompts.
    Uses extracted patterns, conversation context to create optimal responses.
    For agent improvement: generates handling scripts for objections, etc.
    """
    
    def __init__(self):
        """
        Initialize the generator with OpenAI.
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4",
            temperature=0.5,
            max_tokens=500
        )
        logger.info("ScriptGenerator initialized with OpenAI")
    
    async def generate_script(self, context: str, patterns: Optional[List[Dict[str, Any]]] = None, scenario: str = "objection_handling") -> str:
        """
        Generate a script or response for a given scenario.
        
        :param context: Conversation context or objection.
        :param patterns: Optional list of extracted patterns.
        :param scenario: Type like 'objection_handling', 'pa_request'.
        :return: Generated script text.
        """
        prompt = self._build_generation_prompt(context, patterns, scenario)
        
        try:
            response = await self.llm.agenerate([prompt])
            result = response.generations[0][0].text
            script = self._parse_llm_response(result)
            logger.info("Script generated")
            return script
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            raise
    
    def _build_generation_prompt(self, context: str, patterns: Optional[List[Dict[str, Any]]], scenario: str) -> str:
        """
        Build prompt for LLM script generation.
        
        :param context: Context text.
        :param patterns: Optional patterns list.
        :param scenario: Scenario type.
        :return: Formatted prompt.
        """
        patterns_str = ""
        if patterns:
            patterns_str = "Use these successful patterns:\n"
            for pattern in patterns:
                patterns_str += f"- {pattern.get('phrase', '')}: {pattern.get('context', '')}\n"
        else:
            patterns_str = "No specific patterns provided."
        
        prompt = f"""
Generate a professional script for a healthcare prior authorization call representative.

Scenario: {scenario}
Context/Objection: {context}

{patterns_str}

Requirements:
- Be empathetic and professional
- Address the specific concern directly
- Use healthcare industry language appropriately
- Keep response concise (2-3 sentences max)
- Focus on patient care and medical necessity

Output only the script text that the representative should say, nothing else.
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> str:
        """
        Parse the script from LLM response (clean and format).
        
        :param response: Raw response string.
        :return: Cleaned script.
        """
        # Clean the response
        script = response.strip()
        
        # Remove quotes if the entire response is quoted
        if script.startswith('"') and script.endswith('"'):
            script = script[1:-1]
        
        # Remove "Rep:" prefix if present
        if script.startswith("Rep:"):
            script = script[4:].strip()
        
        return script

# Sync wrapper for backward compatibility
def generate_script_sync(context: str, patterns: Optional[List[Dict[str, Any]]] = None, scenario: str = "objection_handling") -> str:
    """Synchronous wrapper for generate_script"""
    generator = ScriptGenerator()
    return asyncio.run(generator.generate_script(context, patterns, scenario))

# Example usage (for testing)
if __name__ == "__main__":
    generator = ScriptGenerator()
    try:
        mock_context = "Insurance objects to cost."
        mock_patterns = [{"type": "objection_handling", "phrase": "Discuss assistance programs.", "context": "cost concerns", "outcome": "success"}]
        script = asyncio.run(generator.generate_script(mock_context, mock_patterns))
        print(f"Generated script: {script}")
    except Exception as e:
        print(f"Error: {e}")