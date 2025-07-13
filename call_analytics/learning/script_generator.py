import logging
from typing import Dict, Any, Optional
import requests
import json

from config.settings import settings
from .pattern_extractor import PatternExtractor  # For integration

logger = logging.getLogger(__name__)

class ScriptGenerator:
    """
    Generates scripts/responses using context-aware LLM prompts.
    Uses extracted patterns, conversation context to create optimal responses.
    For agent improvement: generates handling scripts for objections, etc.
    """
    
    def __init__(self, api_url: str = "https://api.openai.com/v1/chat/completions",  # Placeholder
                 api_key: Optional[str] = None):
        """
        Initialize the generator with LLM API details.
        
        :param api_url: LLM API endpoint.
        :param api_key: API key for authentication.
        """
        self.api_url = api_url
        self.api_key = api_key or settings.get("LLM_API_KEY", None)
        if not self.api_key:
            raise ValueError("LLM API key not provided")
        
        logger.info("ScriptGenerator initialized")
    
    def generate_script(self, context: str, patterns: Optional[List[Dict[str, Any]]] = None, scenario: str = "objection_handling") -> str:
        """
        Generate a script or response for a given scenario.
        
        :param context: Conversation context or objection.
        :param patterns: Optional list of extracted patterns.
        :param scenario: Type like 'objection_handling', 'pa_request'.
        :return: Generated script text.
        """
        prompt = self._build_generation_prompt(context, patterns, scenario)
        
        try:
            response = self._call_llm(prompt)
            script = self._parse_llm_response(response)
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
        patterns_str = json.dumps(patterns, indent=2) if patterns else "No patterns provided."
        prompt = f"""
Generate a script or response for a healthcare prior authorization call.
Scenario: {scenario}
Context: {context}
Use these patterns if relevant: {patterns_str}

Output the script as a string, e.g., "Rep: [response]."
Make it concise, professional, and effective based on success patterns.
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API.
        
        :param prompt: Prompt to send.
        :return: LLM response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4",  # Placeholder
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response: str) -> str:
        """
        Parse the script from LLM response (strip quotes if needed).
        
        :param response: Raw response string.
        :return: Cleaned script.
        """
        return response.strip().strip('"')

# Example usage (for testing)
if __name__ == "__main__":
    generator = ScriptGenerator(api_key="your_api_key_here")
    try:
        mock_context = "Insurance objects to cost."
        mock_patterns = [{"type": "objection_handling", "phrase": "Discuss assistance programs."}]
        script = generator.generate_script(mock_context, mock_patterns)
        print(f"Generated script: {script}")
    except Exception as e:
        print(f"Error: {e}")