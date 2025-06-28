import sys
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from .telephony import make_call
from .conversation import process_message
from .speech import transcribe_audio, synthesize_speech
from .database import log_conversation, get_logs
from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/trigger-call")
async def trigger_call(phone_number: str = Form(...)):
    call_sid = make_call(phone_number)
    return {"status": "Call triggered", "call_sid": call_sid}

@app.get("/logs")
async def fetch_logs():
    return await get_logs()

@app.post("/voice")
async def voice_webhook(request: Request):
    print("Received /voice webhook")
    form = await request.form()
    print("Form data:", form)
    call_sid = form.get("CallSid")
    audio_url = form.get("RecordingUrl")
    print("CallSid:", call_sid, "RecordingUrl:", audio_url)

    if audio_url:
        print("Processing audio_url")
        transcription = await transcribe_audio(audio_url)
        print("Transcription:", transcription)
        response_text, extracted_data = await process_message(transcription)
        print("Response:", response_text, "Extracted data:", extracted_data)
        if "escalate" in response_text.lower():
            twiml = VoiceResponse()
            twiml.dial(os.getenv("HUMAN_ESCALATION_NUMBER"))
            await log_conversation(call_sid, transcription, response_text, extracted_data, escalated=True)
        else:
            speech_url = await synthesize_speech(response_text)
            twiml = VoiceResponse()
            twiml.play(speech_url)
            await log_conversation(call_sid, transcription, response_text, extracted_data)
        return str(twiml)
    else:
        print("No audio_url, sending welcome message")
        twiml = VoiceResponse()
        twiml.say("Welcome to Alfons, your prior authorization assistant. Please provide patient details.")
        twiml.record(max_length=30, action="/voice")
        return str(twiml)