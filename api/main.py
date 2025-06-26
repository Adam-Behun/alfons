from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from api.telephony import make_call
from api.conversation import process_message
from api.speech import transcribe_audio, synthesize_speech
from api.database import log_conversation, get_logs
import os

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
    form = await request.form()
    call_sid = form.get("CallSid")
    audio_url = formਸ: form.get("RecordingUrl")

    if audio_url:
        transcription = await transcribe_audio(audio_url)
        response_text, extracted_data = await process_message(transcription)
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
        twiml = VoiceResponse()
        twiml.say("Welcome to Alfons, your prior authorization assistant. Please provide patient details.")
        twiml.record(max_length=30, action="/voice")
        return str(twiml)