"""
database.py

Module manages all interactions with the Supabase database for the Alfons backend.
It provides functions to log conversations and retrieve conversation logs.

https://supabase.com/docs

Environment variables required:
- SUPABASE_URL: The Supabase project URL
- SUPABASE_KEY: The Supabase service role or anon key

Table used:
- conversations: Stores call and conversation details
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client

load_dotenv()
# Create a Supabase client instance using environment variables
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

async def log_conversation(
        # sample data points to collect during the conversation
        call_sid: str, 
        user_input: str, 
        bot_response: str, 
        extracted_data: dict, 
        escalated: bool = False
):
    data = {
        "call_sid": call_sid,
        "user_input": user_input,
        "bot_response": bot_response,
        "patient_id": extracted_data.get("patient_id"),
        "procedure_code": extracted_data.get("procedure_code"),
        "insurance": extracted_data.get("insurance"),
        "approval_status": extracted_data.get("approval_status"),
        "auth_number": extracted_data.get("auth_number"),
        "escalated": escalated,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Insert the conversation data into the 'conversations' table
    supabase.table("conversations").insert(data).execute()

async def get_logs():
    response = supabase.table("conversations").select("*").execute()
    return response.data