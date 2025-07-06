from langchain_openai import ChatOpenAI
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
    approval_status: str = None
    auth_number: str = None

async def process_message(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process user message using OpenAI GPT-4o-mini and extract relevant data
    Returns tuple of (response_text, extracted_data)
    """
    logger.info(f"Processing message: {text[:100]}...")
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found")
        return "I'm sorry, I'm having trouble accessing my AI services right now.", {
            "patient_id": None,
            "procedure_code": None,
            "insurance": None,
            "approval_status": None,
            "auth_number": None
        }
    
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini-2024-07-18",  # GPT-4o mini - optimal for healthcare conversations
            temperature=0.3,
            max_tokens=500
        )
        
        # Create a voice-optimized prompt for healthcare conversations
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
# Identity
You are Alfons, a professional healthcare prior authorization specialist calling on behalf of a medical provider. You have 5+ years of experience handling insurance authorizations and understand medical terminology, CPT codes, and insurance processes.

# Task
You are calling insurance companies to obtain prior authorization approvals for patient procedures. Your goal is to efficiently gather authorization details, confirm approvals, and document authorization numbers.

# Demeanor
Professional, patient, and persistent. You understand insurance representatives are busy and respect their time while ensuring you get the information needed for patient care.

# Tone
Polite and authoritative - you speak with the confidence of a healthcare administrator who handles these calls daily.

# Level of Enthusiasm
Calm and measured - you're focused on the business at hand without being overly energetic.

# Level of Formality
Professional business language appropriate for B2B healthcare communications.

# Level of Emotion
Matter-of-fact with occasional empathy when discussing patient care needs.

# Filler Words
Occasionally use "um" or "let me see" when processing information to sound natural.

# Pacing
Moderate speed - clear enough for accurate information exchange, but efficient for business calls.

# Instructions
- If you receive a patient ID, procedure code, or authorization number, always repeat it back using phonetic clarity (e.g., "That's authorization A as in Alpha, B as in Bravo, one-two-three")
- If the representative corrects any detail, acknowledge immediately: "I understand, let me correct that"
- Always ask for confirmation of critical details before proceeding
- Keep responses under 25 words for phone efficiency
- Use healthcare terminology appropriately (CPT codes, prior auth, member ID, etc.)

Insurance Rep Response: {text}

Extract information and respond professionally in this exact format:
RESPONSE: [Your professional response to continue the authorization conversation]
PATIENT_ID: [extracted patient/member ID or "none"]
PROCEDURE_CODE: [extracted procedure/CPT code or "none"]  
INSURANCE: [insurance company name or "none"]
APPROVAL_STATUS: [approved/denied/pending/none]
AUTH_NUMBER: [authorization number if provided or "none"]
ESCALATE: [yes/no - if you need human intervention]

Conversation Flow:
- If no patient info yet: Ask for member ID and procedure code
- If info provided: Confirm details using phonetic spelling
- If approved: Request and confirm authorization number
- If denied: Ask for denial reason and reference number
- If unclear: Ask representative to repeat or clarify

Examples of professional responses:
- "Good morning, this is Alfons calling for prior authorization. I have member ID A-B-1-2-3-4 for CPT code 99213. Can you assist?"
- "Let me confirm that authorization number: A as in Alpha, U-T-H, dash, 7-8-9-1-2-3. Is that correct?"
- "I understand this requires additional documentation. What specific information do you need?"
"""
        )
        
        # Generate response
        formatted_prompt = prompt.format(text=text)
        logger.info("Sending request to OpenAI...")
        
        response = await llm.ainvoke(formatted_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Raw AI response: {result}")
        
        # Parse the structured response
        response_text, extracted_data = parse_ai_response(result)
        
        logger.info(f"Parsed response: {response_text}")
        logger.info(f"Extracted data: {extracted_data}")
        
        return response_text, extracted_data
        
    except Exception as e:
        logger.error(f"Error in process_message: {str(e)}")
        
        # Fallback response
        fallback_response = "I understand you're calling about a prior authorization. Could you please provide the patient ID, procedure code, and insurance information?"
        fallback_data = {
            "patient_id": None,
            "procedure_code": None,
            "insurance": None,
            "approval_status": None,
            "auth_number": None
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
        approval_status = None
        auth_number = None
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
        
        approval_status_match = re.search(r'APPROVAL_STATUS:\s*(.+?)(?=\n|$)', response_text)
        if approval_status_match:
            approval_status = approval_status_match.group(1).strip()
            if approval_status.lower() == "none":
                approval_status = None
        
        auth_number_match = re.search(r'AUTH_NUMBER:\s*(.+?)(?=\n|$)', response_text)
        if auth_number_match:
            auth_number = auth_number_match.group(1).strip()
            if auth_number.lower() == "none":
                auth_number = None
        
        escalate_match = re.search(r'ESCALATE:\s*(.+?)(?=\n|$)', response_text)
        if escalate_match:
            escalate_text = escalate_match.group(1).strip().lower()
            escalate = escalate_text in ["yes", "true", "1"]
        
        # If escalation is needed, modify the response
        if escalate:
            response = "I understand this requires additional review. Let me transfer you to my supervisor who can better assist with this authorization."
        
        # Ensure response is concise for phone conversations
        if len(response.split()) > 25:
            # Truncate overly long responses for phone efficiency
            words = response.split()[:25]
            response = " ".join(words) + "..."
        
        extracted_data = {
            "patient_id": patient_id,
            "procedure_code": procedure_code,
            "insurance": insurance,
            "approval_status": approval_status,
            "auth_number": auth_number,
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
            "approval_status": None,
            "auth_number": None,
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
        "insurance": None,
        "approval_status": None,
        "auth_number": None
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
    
    # Common patterns for approval status
    approval_patterns = [
        r'(?:approved|denied|pending|authorization)[\s:]*([A-Za-z]+)',
        r'status[\s:]*([A-Za-z]+)'
    ]
    
    # Common patterns for authorization number
    auth_patterns = [
        r'authorization\s+(?:number|code)[\s:]*([A-Z0-9\-]+)',
        r'auth[\s:]*([A-Z0-9\-]+)',
        r'reference[\s:]*([A-Z0-9\-]+)'
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
    
    # Extract approval status
    for pattern in approval_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted["approval_status"] = match.group(1).lower()
            break
    
    # Extract authorization number
    for pattern in auth_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted["auth_number"] = match.group(1).upper()
            break
    
    return extracted