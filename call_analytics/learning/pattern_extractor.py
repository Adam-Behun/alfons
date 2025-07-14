import logging
from typing import Dict, List, Any, Optional
import json
import asyncio
from langchain_openai import ChatOpenAI

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class PatternExtractor:
    """
    Extracts patterns using OpenAI: winning phrases, objection handling, success patterns.
    Analyzes transcripts or conversation flows to identify repeatable learnings.
    Outputs structured patterns for storage in memory.
    """
    
    def __init__(self):
        """
        Initialize the extractor with OpenAI.
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4",
            temperature=0.3,
            max_tokens=1000
        )
        logger.info("PatternExtractor initialized with OpenAI")
    
    async def extract_patterns(self, transcript: str, analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract patterns from transcript and optional analysis.
        
        :param transcript: Full transcript text.
        :param analysis: Optional dict from ConversationAnalyzer.
        :return: List of patterns {'type': str, 'phrase': str, 'context': str, 'outcome': str}.
        """
        prompt = self._build_extraction_prompt(transcript, analysis)
        
        try:
            response = await self.llm.agenerate([prompt])
            result = response.generations[0][0].text
            patterns = self._parse_llm_response(result)
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

Output ONLY a valid JSON list in this exact format:
[{{"type": "winning_phrase", "phrase": "example phrase", "context": "when used during cost objection", "outcome": "success"}}, {{"type": "objection_handling", "phrase": "let me explain the benefits", "context": "eligibility concerns", "outcome": "success"}}]

Types must be one of: winning_phrase, objection_handling, success_pattern, failure_pattern.
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON list from LLM response.
        
        :param response: Raw response string.
        :return: List of pattern dicts.
        """
        try:
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            patterns = json.loads(response)
            
            # Validate structure
            if not isinstance(patterns, list):
                raise ValueError("Response is not a list")
            
            for pattern in patterns:
                if not all(key in pattern for key in ['type', 'phrase', 'context', 'outcome']):
                    raise ValueError("Pattern missing required fields")
            
            return patterns
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in LLM response: {response}")
            return []  # Return empty list instead of failing
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

# Sync wrapper for backward compatibility
def extract_patterns_sync(transcript: str, analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Synchronous wrapper for extract_patterns"""
    extractor = PatternExtractor()
    return asyncio.run(extractor.extract_patterns(transcript, analysis))

# Example usage (for testing)
if __name__ == "__main__":
    extractor = PatternExtractor()
    try:
        mock_transcript = "Rep: Here's why we need it. Insurance: Approved."
        patterns = asyncio.run(extractor.extract_patterns(mock_transcript))
        print(json.dumps(patterns, indent=2))
    except Exception as e:
        print(f"Error: {e}")