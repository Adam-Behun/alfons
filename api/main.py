"""
main.py

This is the entry point for the Alfons backend API, built with FastAPI. 

FastAPI Overview:
    https://fastapi.tiangolo.com/
    1. Built on Starlette, .py framework supporting asynchronous programming - multi-tasking, concurrency.
    2. Pydantic and Type Hints to Auto-Generate OpenAPI Docs (e.g., age: int ensures age is an integer)
    3. Without type hints, invalid data (e.g., int instead str in Twilio webhook) crash app at runtime
    4. Async/Await: functions (/voice endpoint) run without blocking other tasks. async def marks a function as asynchronous, and await pauses it until a task (e.g., querying Supabase) completes.

main.py is responsible for setting up the API routes (endpoints) and handling HTTP requests and 
responses.
The actual work (business logic)—such as making phone calls, processing messages, 
transcribing audio, or interacting with the database—is handled by separate Python files 
(helper modules) like telephony.py, conversation.py, speech.py, and database.py.
"""

# import standart and thrid-party libraries
import sys
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware

# import local modules
from .telephony import make_call
from .conversation import process_message
from .speech import transcribe_audio, synthesize_speech
from .database import log_conversation, get_logs
from twilio.twiml.voice_response import VoiceResponse

# Initialize FastAPI app
app = FastAPI()

# Ensures Next.js app can call the FastAPI backend locally without browser errors.
# ok for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # (use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to trigger an outbound call
# Expects a phone number as form data and calls the make_call function in telephony.py
@app.post("/trigger-call")
async def trigger_call(phone_number: str = Form(...)):
    call_sid = make_call(phone_number)
    return {"status": "Call triggered", "call_sid": call_sid}

@app.get("/logs")
async def fetch_logs():
    return await get_logs()

@app.post("/voice")
async def voice_webhook(request: Request):

    """
    Webhook endpoint for Twilio to handle voice call events.
    Handles incoming recordings, processes them, and responds with TwiML.
    """

    form = await request.form()
    call_sid = form.get("CallSid")
    audio_url = form.get("RecordingUrl")

    if audio_url:
        # If a recording is present, transcribe it and process the message
        transcription = await transcribe_audio(audio_url)
        response_text, extracted_data = await process_message(transcription)
        
        if "escalate" in response_text.lower():
            # Escalate to a human  using number set in environment variable
            twiml = VoiceResponse()
            twiml.dial(os.getenv("HUMAN_ESCALATION_NUMBER"))
            await log_conversation(call_sid, transcription, response_text, extracted_data, escalated=True)
        else:
            # If not escalating, synthesize the response and play it back
            speech_url = await synthesize_speech(response_text)
            twiml = VoiceResponse()
            twiml.play(speech_url)
            await log_conversation(call_sid, transcription, response_text, extracted_data)
        return str(twiml)
    else:
        # If no recording, prompt the caller for input
        twiml = VoiceResponse()
        twiml.say("Welcome to Alfons, your prior authorization assistant. Please provide patient details.")
        twiml.record(max_length=30, action="/voice")
        return str(twiml)