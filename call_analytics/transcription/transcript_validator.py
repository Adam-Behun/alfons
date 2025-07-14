import logging
from typing import Dict, List, Any, Optional
import json
import asyncio
from langchain_openai import ChatOpenAI

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class TranscriptValidator:
    """
    OpenAI-based validation for transcripts.
    Checks for errors, coherence, medical jargon accuracy.
    Uses OpenAI GPT-4 for validation tasks.
    """
    
    def __init__(self):
        """
        Initialize the validator with OpenAI.
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4",
            temperature=0.2,  # Lower for validation accuracy
            max_tokens=800
        )
        logger.info("TranscriptValidator initialized with OpenAI")
    
    async def validate(self, transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the transcript using LLM.
        
        :param transcript: Full transcript text.
        :param segments: List of segment dicts.
        :return: Validation result {'valid': bool, 'errors': list, 'corrections': dict}.
        """
        prompt = self._build_validation_prompt(transcript, segments)
        
        try:
            response = await self.llm.agenerate([prompt])
            result = response.generations[0][0].text
            validation = self._parse_llm_response(result)
            logger.info("Transcript validation completed")
            return validation
        except Exception as e:
            logger.error(f"Error validating transcript: {e}")
            raise
    
    def _build_validation_prompt(self, transcript: str, segments: List[Dict[str, Any]]) -> str:
        """
        Build prompt for LLM validation.
        
        :param transcript: Transcript text.
        :param segments: Segments for context.
        :return: Formatted prompt.
        """
        # Limit segments to prevent prompt overflow
        segment_preview = segments[:5] if len(segments) > 5 else segments
        segment_str = json.dumps(segment_preview, indent=2)
        
        prompt = f"""
Validate this healthcare prior authorization call transcript for accuracy and coherence.

Check for:
1. Obvious transcription errors (misheard words)
2. Medical terminology accuracy
3. Logical conversation flow
4. Speaker attribution consistency

Transcript: {transcript}

Sample segments: {segment_str}

Respond with ONLY valid JSON in this format:
{{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "errors": ["specific error descriptions"],
  "corrections": {{"original_phrase": "corrected_phrase"}},
  "medical_terms_check": "passed/failed",
  "coherence_score": 0.0-1.0
}}

Be conservative - only flag clear errors, not stylistic issues.
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        :param response: Raw response string.
        :return: Parsed validation dict.
        """
        try:
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            validation = json.loads(response)
            
            # Ensure required fields with defaults
            default_validation = {
                "valid": True,
                "confidence": 0.8,
                "errors": [],
                "corrections": {},
                "medical_terms_check": "passed",
                "coherence_score": 0.8
            }
            
            # Merge with defaults
            for key, default_value in default_validation.items():
                if key not in validation:
                    validation[key] = default_value
            
            return validation
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in validation response: {response}")
            # Return default "valid" response on parse error
            return {
                "valid": True,
                "confidence": 0.5,
                "errors": ["Could not parse validation response"],
                "corrections": {},
                "medical_terms_check": "unknown",
                "coherence_score": 0.5
            }
        except Exception as e:
            logger.error(f"Error parsing validation response: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "errors": [f"Validation error: {str(e)}"],
                "corrections": {},
                "medical_terms_check": "failed",
                "coherence_score": 0.0
            }

# Sync wrapper for backward compatibility
def validate_transcript_sync(transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper for validate"""
    validator = TranscriptValidator()
    return asyncio.run(validator.validate(transcript, segments))

# Example usage (for testing)
if __name__ == "__main__":
    validator = TranscriptValidator()
    try:
        # Test validate (mock data)
        mock_transcript = "Doctor: We need prior auth for XYZ drug."
        mock_segments = [{"text": "We need prior auth for XYZ drug.", "start": 0, "end": 5}]
        result = asyncio.run(validator.validate(mock_transcript, mock_segments))
        print(f"Validation: {result}")
    except Exception as e:
        print(f"Error: {e}")