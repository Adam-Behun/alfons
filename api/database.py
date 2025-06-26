from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

async def log_conversation(call_sid: str, user_input: str, bot_response: str, extracted_data: dict, escalated: bool = False):
    data = {
        "call_sid": call_sid,
        "user_input": user_input,
        "bot_response": bot_response,
        "patient_id": extracted_data.get("patient_id"),
        "procedure_code": extracted_data.get("procedure_code"),
        "insurance": extracted_data.get("insurance"),
        "escalated": escalated,
        "timestamp": datetime.utcnow().isoformat()
    }
    supabase.table("conversations").insert(data).execute()

async def get_logs():
    response = supabase.table("conversations").select("*").execute()
    return response.data