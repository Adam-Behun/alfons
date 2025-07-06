"""
database.py

Module manages all interactions with the Supabase database for the Alfons backend.
It provides functions to log conversations and retrieve conversation logs with authorization data support.

https://supabase.com/docs

Environment variables required:
- SUPABASE_URL: The Supabase project URL
- SUPABASE_KEY: The Supabase service role or anon key

Table used:
- conversations: Stores call and conversation details including authorization data
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Create a Supabase client instance using environment variables
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def validate_authorization_data(extracted_data: dict) -> dict:
    """
    Validate and clean extracted authorization data
    
    Args:
        extracted_data: Raw extracted data dictionary
        
    Returns:
        Cleaned and validated data dictionary
    """
    # Define valid approval statuses
    valid_statuses = ['approved', 'denied', 'pending', 'under_review', 'requires_documentation']
    
    # Clean and validate approval status
    approval_status = extracted_data.get("approval_status")
    if approval_status:
        approval_status = approval_status.lower().strip()
        if approval_status not in valid_statuses:
            # Try to map common variations
            status_mappings = {
                'approve': 'approved',
                'deny': 'denied',
                'reject': 'denied',
                'rejected': 'denied',
                'wait': 'pending',
                'waiting': 'pending',
                'review': 'under_review',
                'reviewing': 'under_review'
            }
            approval_status = status_mappings.get(approval_status, approval_status)
    
    # Clean authorization number
    auth_number = extracted_data.get("auth_number")
    if auth_number:
        # Remove common prefixes and clean format
        auth_number = auth_number.upper().strip()
        if auth_number.startswith(('AUTH-', 'AUTH_', 'AUTH ')):
            auth_number = auth_number[5:]
        elif auth_number.startswith('AUTH'):
            auth_number = auth_number[4:]
    
    # Clean patient ID
    patient_id = extracted_data.get("patient_id")
    if patient_id:
        patient_id = patient_id.strip().upper()
    
    # Clean procedure code
    procedure_code = extracted_data.get("procedure_code")
    if procedure_code:
        procedure_code = procedure_code.strip().upper()
        # Remove common prefixes
        if procedure_code.startswith(('CPT-', 'CPT_', 'CPT ')):
            procedure_code = procedure_code[4:]
        elif procedure_code.startswith('CPT'):
            procedure_code = procedure_code[3:]
    
    return {
        "patient_id": patient_id,
        "procedure_code": procedure_code,
        "insurance": extracted_data.get("insurance"),
        "approval_status": approval_status,
        "auth_number": auth_number,
        "escalate": extracted_data.get("escalate", False)
    }

async def log_conversation(
        call_sid: str, 
        user_input: str, 
        bot_response: str, 
        extracted_data: dict, 
        escalated: bool = False
):
    """
    Log conversation data to Supabase including authorization details
    
    Args:
        call_sid: Twilio call identifier
        user_input: Transcribed user speech
        bot_response: AI generated response
        extracted_data: Dictionary containing extracted fields
        escalated: Whether call was escalated to human
    """
    # Validate and clean the extracted data
    validated_data = validate_authorization_data(extracted_data)
    
    data = {
        "call_sid": call_sid,
        "user_input": user_input,
        "bot_response": bot_response,
        "patient_id": validated_data.get("patient_id"),
        "procedure_code": validated_data.get("procedure_code"),
        "insurance": validated_data.get("insurance"),
        "approval_status": validated_data.get("approval_status"),
        "auth_number": validated_data.get("auth_number"),
        "escalated": escalated,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Insert the conversation data into the 'conversations' table
        result = supabase.table("conversations").insert(data).execute()
        logger.info(f"Successfully logged conversation for call {call_sid}")
        logger.debug(f"Logged data: {data}")
        return result
    except Exception as e:
        logger.error(f"Failed to log conversation for call {call_sid}: {str(e)}")
        raise

async def get_logs():
    """
    Retrieve all conversation logs from the database
    
    Returns:
        List of conversation log dictionaries
    """
    try:
        response = supabase.table("conversations").select("*").order("timestamp", desc=True).execute()
        logger.info(f"Retrieved {len(response.data)} conversation logs")
        return response.data
    except Exception as e:
        logger.error(f"Failed to retrieve conversation logs: {str(e)}")
        raise