import logging
from typing import Dict, List, Any, Optional
import requests
import json

from config.settings import settings
from ..analytics.conversation_analyzer import ConversationAnalyzer  # For integration

logger = logging.getLogger(__name__)

class PatternExtractor:
    """
    Extracts patterns using LLM: winning phrases, objection handling, success patterns.
    Analyzes transcripts or conversation flows to identify repeatable learnings.
    Outputs structured patterns for storage in memory.
    """
    
    def __init__(self, api_url: str = "https://api.openai.com/v1/chat/completions",  # Placeholder
                 api_key: Optional[str] = None):
        """
        Initialize the extractor with LLM API details.
        
        :param api_url: LLM API endpoint.
        :param api_key: API key for authentication.
        """
        self.api_url = api_url
        self.api_key = api_key or settings.get("LLM_API_KEY", None)
        if not self.api_key:
            raise ValueError("LLM API key not provided")
        
        logger.info("PatternExtractor initialized")
    
    def extract_patterns(self, transcript: str, analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract patterns from transcript and optional analysis.
        
        :param transcript: Full transcript text.
        :param analysis: Optional dict from ConversationAnalyzer.
        :return: List of patterns {'type': str, 'phrase': str, 'context': str, 'outcome': str}.
        """
        prompt = self._build_extraction_prompt(transcript, analysis)
        
        try:
            response = self._call_llm(prompt)
            patterns = self._parse_llm_response(response)
            logger.info(f"Extracted {len(patterns)} patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            raise
    
    def _build_extraction_prompt(self, transcript: str, analysis: Optional[Dict[str, Any]]) -> str:
        """
        Build prompt for LLM pattern extraction.
        
        :param transcript: Transcript text.
        :param analysis: Optional analysis dict.
        :return: Formatted prompt.
        """
        analysis_str = json.dumps(analysis, indent=2) if analysis else "No additional analysis provided."
        prompt = f"""
Analyze the following healthcare prior authorization call transcript to extract patterns:
- Winning phrases that led to success.
- Effective objection handling.
- Common success/failure patterns (e.g., timing, phrasing).
- Scripts or sequences that worked well.

Transcript: {transcript}
Analysis: {analysis_str}

Output as JSON list: [{{"type": "winning_phrase", "phrase": "example", "context": "when used", "outcome": "success"}}, ...]
Types: winning_phrase, objection_handling, success_pattern, failure_pattern.
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
            "temperature": 0.5
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON list from LLM response.
        
        :param response: Raw response string.
        :return: List of pattern dicts.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in LLM response")

# Example usage (for testing)
if __name__ == "__main__":
    extractor = PatternExtractor(api_key="your_api_key_here")
    try:
        mock_transcript = "Rep: Here's why we need it. Insurance: Approved."
        patterns = extractor.extract_patterns(mock_transcript)
        print(json.dumps(patterns, indent=2))
    except Exception as e:
        print(f"Error: {e}")