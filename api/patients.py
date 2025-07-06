# Patient database functions for Alfons Prior Authorization Bot
# Handles all patient data operations with Supabase

import os
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

async def get_patients():
    """Retrieve all patients from the database"""
    response = supabase.table("patients").select("*").order("appointment_time").execute()
    return response.data

async def get_patient_by_id(patient_id: int):
    """Retrieve a specific patient by ID"""
    response = supabase.table("patients").select("*").eq("id", patient_id).execute()
    return response.data[0] if response.data else None

async def update_patient_auth_status(patient_id: int, status: str):
    """Update the prior authorization status for a patient"""
    data = {
        "prior_auth_status": status,
        "updated_at": datetime.utcnow().isoformat()
    }
    response = supabase.table("patients").update(data).eq("id", patient_id).execute()
    return response.data

async def get_pending_authorizations():
    """Get all patients with pending prior authorization status"""
    response = supabase.table("patients").select("*").eq("prior_auth_status", "Pending").order("appointment_time").execute()
    return response.data