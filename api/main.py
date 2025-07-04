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

"""
main.py - Improved version with better error handling and reliability
"""

import sys
import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules with error handling
try:
    from .telephony import make_call
    from .conversation import process_message
    from .speech import transcribe_audio, synthesize_speech
    from .database import log_conversation, get_logs
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    # For development, try relative imports
    try:
        from telephony import make_call
        from conversation import process_message
        from speech import transcribe_audio, synthesize_speech
        from database import log_conversation, get_logs
    except ImportError as e2:
        logger.error(f"Failed relative imports too: {e2}")
        raise

# Initialize FastAPI app
app = FastAPI(title="Alfons Prior Authorization Bot", version="1.0.0")

# Create static directory if it doesn't exist
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# Add static file serving
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Validate required environment variables on startup
REQUIRED_ENV_VARS = [
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN", 
    "TWILIO_PHONE_NUMBER",
    "BASE_URL",
    "ELEVENLABS_API_KEY",
    "XAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY"
]

@app.on_event("startup")
async def startup_event():
    """Validate environment variables on startup"""
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    logger.info("All required environment variables are set")
    logger.info(f"BASE_URL: {os.getenv('BASE_URL')}")
    logger.info(f"Static directory created: {os.path.abspath(static_dir)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "Alfons API is running",
        "version": "1.0.0",
        "endpoints": {
            "trigger_call": "/trigger-call",
            "voice_webhook": "/voice",
            "logs": "/logs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "base_url": os.getenv("BASE_URL"),
        "static_dir_exists": os.path.exists(static_dir),
        "twilio_configured": bool(os.getenv("TWILIO_ACCOUNT_SID")),
        "elevenlabs_configured": bool(os.getenv("ELEVENLABS_API_KEY")),
        "xai_configured": bool(os.getenv("XAI_API_KEY")),
        "supabase_configured": bool(os.getenv("SUPABASE_URL"))
    }

@app.post("/trigger-call")
async def trigger_call(phone_number: str = Form(...)):
    """Trigger an outbound call"""
    logger.info(f"Triggering call to: {phone_number}")
    
    # Validate phone number format
    if not phone_number.startswith('+'):
        raise HTTPException(status_code=400, detail="Phone number must start with +")
    
    try:
        call_sid = make_call(phone_number)
        logger.info(f"Call triggered successfully: {call_sid}")
        return {"status": "success", "call_sid": call_sid, "phone_number": phone_number}
    except Exception as e:
        logger.error(f"Error triggering call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger call: {str(e)}")

@app.get("/logs")
async def fetch_logs():
    """Fetch conversation logs"""
    try:
        logs = await get_logs()
        logger.info(f"Fetched {len(logs)} logs")
        return logs
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")

@app.get("/voice")
async def voice_get():
    """Test endpoint to verify webhook is accessible"""
    return {
        "status": "Voice webhook is accessible",
        "message": "This endpoint handles Twilio voice webhooks via POST"
    }

@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Webhook endpoint for Twilio to handle voice call events.
    This is the main entry point for all voice interactions.
    """
    logger.info("VOICE WEBHOOK CALLED")
    
    try:
        # Parse form data from Twilio
        form = await request.form()
        form_dict = dict(form)
        logger.info(f"Form data received: {form_dict}")
        
        call_sid = form.get("CallSid")
        audio_url = form.get("RecordingUrl")
        call_status = form.get("CallStatus")
        
        logger.info(f"Call SID: {call_sid}")
        logger.info(f"Audio URL: {audio_url}")
        logger.info(f"Call Status: {call_status}")

        # Create TwiML response
        twiml = VoiceResponse()

        if audio_url:
            # We have a recording to process
            logger.info("Processing audio recording...")
            
            try:
                # Step 1: Transcribe the audio
                logger.info("Transcribing audio...")
                transcription = await transcribe_audio(audio_url)
                logger.info(f"Transcription: {transcription}")
                
                if transcription == "[Error during transcription]":
                    twiml.say("I'm sorry, I couldn't understand your response. Please try again.")
                    twiml.record(action=f"{os.getenv('BASE_URL')}/voice", maxLength="30", playBeep=True)
                    return Response(content=str(twiml), media_type="application/xml")
                
                # Step 2: Process the message with AI
                logger.info("Processing message with AI...")
                response_text, extracted_data = await process_message(transcription)
                logger.info(f"Bot response: {response_text}")
                logger.info(f"Extracted data: {extracted_data}")
                
                # Step 3: Check if we need to escalate
                if "escalate" in response_text.lower():
                    logger.info("Escalating to human...")
                    human_number = os.getenv("HUMAN_ESCALATION_NUMBER")
                    if human_number:
                        twiml.say("I'm transferring you to a human representative. Please hold.")
                        twiml.dial(human_number)
                    else:
                        twiml.say("I'd like to transfer you to a human representative, but none are available right now. Please call back later.")
                    
                    await log_conversation(call_sid, transcription, response_text, extracted_data, escalated=True)
                else:
                    # Step 4: Generate speech response
                    logger.info("Generating speech response...")
                    speech_url = await synthesize_speech(response_text)
                    
                    if speech_url:
                        logger.info(f"Playing generated speech: {speech_url}")
                        twiml.play(speech_url)
                    else:
                        logger.info("Using TTS fallback")
                        twiml.say(response_text)
                    
                    # Continue conversation
                    twiml.pause(length=1)
                    twiml.say("Is there anything else I can help you with?")
                    twiml.record(action=f"{os.getenv('BASE_URL')}/voice", maxLength="30", playBeep=True)
                    
                    await log_conversation(call_sid, transcription, response_text, extracted_data)
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                twiml.say("I'm sorry, I'm having trouble processing your request. Let me try again.")
                twiml.record(action=f"{os.getenv('BASE_URL')}/voice", maxLength="30", playBeep=True)
        
        else:
            # Initial call - no recording yet
            logger.info("Initial call - prompting for input...")
            twiml.say("Welcome to Alfons, your prior authorization assistant.")
            twiml.pause(length=1)
            twiml.say("Please tell me your patient ID, procedure code, and insurance information after the beep.")
            
            action_url = f"{os.getenv('BASE_URL')}/voice"
            logger.info(f"Using action URL: {action_url}")
            
            twiml.record(action=action_url, maxLength="30", playBeep=True)
        
        twiml_str = str(twiml)
        logger.info(f"Returning TwiML: {twiml_str}")
        return Response(content=twiml_str, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Critical error in voice webhook: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Always return valid TwiML to prevent Twilio errors
        emergency_twiml = VoiceResponse()
        emergency_twiml.say("I apologize, but I'm experiencing technical difficulties. Please try calling again in a few minutes.")
        
        return Response(content=str(emergency_twiml), media_type="application/xml")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)