from langchain_xai import ChatXAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
import json
import re
import logging
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class AuthorizationData(BaseModel):
    patient_id: str = None
    procedure_code: str = None
    insurance: str = None

async def process_message(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process user message using XAI/Grok and extract relevant data
    Returns tuple of (response_text, extracted_data)
    """
    logger.info(f"Processing message: {text[:100]}...")
    
    # Check if XAI API key is available
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        logger.error("XAI API key not found")
        return "I'm sorry, I'm having trouble accessing my AI services right now.", {
            "patient_id": None,
            "procedure_code": None,
            "insurance": None
        }
    
    try:
        # Initialize the LLM
        llm = ChatXAI(
            api_key=xai_api_key,
            model="grok-3-mini",
            temperature=0.3,  # Lower temperature for more consistent responses
            max_tokens=500
        )
        
        # Create a more structured prompt
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
You are Alfons, a helpful prior authorization assistant. Your job is to:
1. Extract patient information (patient ID, procedure code, insurance)
2. Respond empathetically and professionally
3. Determine if the request needs human escalation

Patient Input: {text}

Please respond in this exact format:
RESPONSE: [Your empathetic response to the patient]
PATIENT_ID: [extracted patient ID or "none"]
PROCEDURE_CODE: [extracted procedure code or "none"]  
INSURANCE: [extracted insurance name or "none"]
ESCALATE: [yes/no - whether this needs human escalation]

Rules:
- If you can't extract clear information, ask for clarification
- If the request is complex, urgent, or involves a complaint, set ESCALATE to "yes"
- Always be empathetic and professional
- Keep responses concise but helpful
"""
        )
        
        # Generate response
        formatted_prompt = prompt.format(text=text)
        logger.info("Sending request to XAI...")
        
        response = await llm.agenerate([formatted_prompt])
        result = response.generations[0][0].text
        
        logger.info(f"Raw AI response: {result}")
        
        # Parse the structured response
        response_text, extracted_data = parse_ai_response(result)
        
        logger.info(f"Parsed response: {response_text}")
        logger.info(f"Extracted data: {extracted_data}")
        
        return response_text, extracted_data
        
    except Exception as e:
        logger.error(f"Error in process_message: {str(e)}")
        
        # Fallback response
        fallback_response = "I understand you're calling about a prior authorization. Could you please provide your patient ID, procedure code, and insurance information?"
        fallback_data = {
            "patient_id": None,
            "procedure_code": None,
            "insurance": None
        }
        
        return fallback_response, fallback_data

def parse_ai_response(response_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse the structured AI response into response text and extracted data
    """
    try:
        # Initialize default values
        response = "I'm here to help with your prior authorization request."
        patient_id = None
        procedure_code = None
        insurance = None
        escalate = False
        
        # Extract using regex patterns
        response_match = re.search(r'RESPONSE:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL)
        if response_match:
            response = response_match.group(1).strip()
        
        patient_id_match = re.search(r'PATIENT_ID:\s*(.+?)(?=\n|$)', response_text)
        if patient_id_match:
            patient_id = patient_id_match.group(1).strip()
            if patient_id.lower() == "none":
                patient_id = None
        
        procedure_code_match = re.search(r'PROCEDURE_CODE:\s*(.+?)(?=\n|$)', response_text)
        if procedure_code_match:
            procedure_code = procedure_code_match.group(1).strip()
            if procedure_code.lower() == "none":
                procedure_code = None
        
        insurance_match = re.search(r'INSURANCE:\s*(.+?)(?=\n|$)', response_text)
        if insurance_match:
            insurance = insurance_match.group(1).strip()
            if insurance.lower() == "none":
                insurance = None
        
        escalate_match = re.search(r'ESCALATE:\s*(.+?)(?=\n|$)', response_text)
        if escalate_match:
            escalate_text = escalate_match.group(1).strip().lower()
            escalate = escalate_text in ["yes", "true", "1"]
        
        # If escalation is needed, modify the response
        if escalate:
            response = "I understand this is an important matter. Let me connect you with a human representative who can better assist you."
        
        extracted_data = {
            "patient_id": patient_id,
            "procedure_code": procedure_code,
            "insurance": insurance,
            "escalate": escalate
        }
        
        return response, extracted_data
        
    except Exception as e:
        logger.error(f"Error parsing AI response: {str(e)}")
        
        # Fallback parsing - just return the raw response
        fallback_data = {
            "patient_id": None,
            "procedure_code": None,
            "insurance": None,
            "escalate": False
        }
        
        return response_text, fallback_data

def extract_info_from_text(text: str) -> Dict[str, Any]:
    """
    Fallback function to extract information using regex patterns
    """
    extracted = {
        "patient_id": None,
        "procedure_code": None,
        "insurance": None
    }
    
    # Common patterns for patient ID
    patient_id_patterns = [
        r'patient\s+(?:id|number)[\s:]*([A-Z0-9]+)',
        r'id[\s:]*([A-Z0-9]+)',
        r'patient[\s:]*([A-Z0-9]+)'
    ]
    
    # Common patterns for procedure codes
    procedure_patterns = [
        r'procedure\s+code[\s:]*([A-Z0-9]+)',
        r'cpt[\s:]*([0-9]+)',
        r'code[\s:]*([A-Z0-9]+)'
    ]
    
    # Common patterns for insurance
    insurance_patterns = [
        r'insurance[\s:]*([A-Za-z\s]+)',
        r'covered\s+by[\s:]*([A-Za-z\s]+)',
        r'plan[\s:]*([A-Za-z\s]+)'
    ]
    
    text_lower = text.lower()
    
    # Extract patient ID
    for pattern in patient_id_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted["patient_id"] = match.group(1).upper()
            break
    
    # Extract procedure code
    for pattern in procedure_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted["procedure_code"] = match.group(1).upper()
            break
    
    # Extract insurance
    for pattern in insurance_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted["insurance"] = match.group(1).title().strip()
            break
    
    return extracted