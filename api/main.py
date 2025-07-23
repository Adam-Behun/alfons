import sys
import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile

from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

# Import core components with dependencies
from .telephony import get_telephony_service, EnhancedTelephonyService
from .speech import get_speech_router, SpeechProcessingRouter
from .conversation import get_conversation_manager, EnhancedConversationManager
from .s2s_pipeline import create_s2s_pipeline, S2SPipeline

# Import analytics from call_analytics (simplified)
from call_analytics.input_handler import InputHandler
from call_analytics.learning_pipeline import MemoryManager
from call_analytics.task_queue import process_call, get_queue_status, health_check as queue_health_check
from call_analytics.mongo_connector import AsyncMongoConnector

# Hooks
from .hooks.learning_hook import handle_call_completion, CallEndPayload

from shared.config import config
from shared.logging import get_logger

# Load env
load_dotenv()

logger = get_logger(__name__)

# Flags
learning_hook_enabled: bool = True  # Assume enabled for MVP
queue_processing_enabled: bool = False  # Initialize to False

# ===== App Configuration =====
# FastAPI app
app = FastAPI(
    title="Alfons Prior Auth Bot",
    version="1.0.0-mvp",
    description="Streamlined S2S prior auth bot"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Dependencies =====
async def get_mongo() -> AsyncMongoConnector:
    mongo = AsyncMongoConnector()
    await mongo.connect()
    yield mongo
    mongo.close_connection()

async def get_input_handler() -> InputHandler:
    return InputHandler()

# ===== Startup/Shutdown Events =====
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Alfons API...")

    # Validate env
    required = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "BASE_URL", "REDIS_URL", "MONGODB_URL"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing env: {missing}")

    # Static dir
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Check queue availability
    global queue_processing_enabled
    try:
        task = queue_health_check.delay()
        result = task.get(timeout=5)
        if result.get('status') == 'healthy':
            queue_processing_enabled = True
            logger.info("Task queue is available and healthy")
        else:
            logger.warning("Task queue health check failed")
    except Exception as e:
        logger.warning(f"Task queue not available - calls will not be processed asynchronously: {str(e)}")

    # Health checks (simplified)
    logger.info("API startup complete")

# ===== Health and Monitoring =====
@app.get("/health")
async def health_check():
    health = {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "services": {}}

    try:
        telephony = get_telephony_service()
        health["services"]["telephony"] = await telephony.get_service_health()

        speech = get_speech_router()
        health["services"]["speech"] = await speech.health_check()

        conv = get_conversation_manager()
        health["services"]["conversation"] = await conv.health_check()

        # Queue if enabled
        if queue_processing_enabled:
            health["services"]["queue"] = queue_health_check.delay().get(timeout=5)

        # Overall
        if any(s.get("status") != "healthy" for s in health["services"].values()):
            health["status"] = "degraded"

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health

# ===== Data Endpoints (Patients/Logs) =====
@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str, mongo: AsyncMongoConnector = Depends(get_mongo)):
    try:
        collection = await mongo.get_collection("patients")
        patient = await collection.find_one({"_id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        patient["id"] = str(patient["_id"])
        del patient["_id"]
        return patient
    except Exception as e:
        logger.error(f"Failed to fetch patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients")
async def get_patients(mongo: AsyncMongoConnector = Depends(get_mongo)):
    try:
        collection = await mongo.get_collection("patients")
        cursor = collection.find({})
        patients = await cursor.to_list(length=100)
        for patient in patients:
            patient["id"] = str(patient["_id"])
            del patient["_id"]
        return patients
    except Exception as e:
        logger.warning(f"Failed to fetch patients: {str(e)}")
        return []  # Fallback to empty array if DB query fails

@app.get("/logs")
async def get_logs(mongo: AsyncMongoConnector = Depends(get_mongo)):
    try:
        collection = await mongo.get_collection("logs")
        cursor = collection.find({}).sort("timestamp", -1)
        logs = await cursor.to_list(length=50)
        for log in logs:
            log["id"] = str(log["_id"])
            del log["_id"]
        return logs
    except Exception as e:
        logger.warning(f"Failed to fetch logs: {str(e)}")
        return []  # Fallback to empty array if DB query fails

# ===== Core API Endpoints (S2S, Conversation, Speech) =====
@app.get("/")
async def root():
    return {
        "status": "Running",
        "version": "1.0.0-mvp",
        "endpoints": {
            "s2s": {
                "trigger_call": "/s2s/trigger-call",
                "media_stream": "/ws/media-stream"
            },
            "conversation": "/conversation/process",
            "speech": "/speech/process",
            "analytics": "/upload-audio",
            "health": "/health"
        }
    }

@app.post("/s2s/trigger-call")
async def trigger_s2s_call(
    phone_number: str = Form(...),
    context: Optional[str] = Form(None),
    telephony: EnhancedTelephonyService = Depends(get_telephony_service)
):
    try:
        call_context = json.loads(context) if context else {}
        call_sid = telephony.make_call(phone_number, call_context)
        return {"status": "success", "call_sid": call_sid}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.websocket("/ws/media-stream")
async def media_stream_ws(websocket: WebSocket, telephony: EnhancedTelephonyService = Depends(get_telephony_service)):
    await websocket.accept()
    try:
        await telephony.handle_stream_connection(websocket, websocket.url.path)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))

@app.post("/conversation/process")
async def process_conversation(
    conversation_id: str = Form(...),
    message: str = Form(...),
    call_sid: Optional[str] = Form(None),
    conv_manager: EnhancedConversationManager = Depends(get_conversation_manager)
):
    try:
        response, extracted = await conv_manager.process_message(conversation_id, message, call_sid)
        return {"response": response, "extracted": extracted}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/speech/process")
async def process_speech(
    audio: UploadFile = File(...),
    context: Optional[str] = Form(None),
    speech_router: SpeechProcessingRouter = Depends(get_speech_router)
):
    try:
        speech_context = json.loads(context) if context else {}
        audio_content = await audio.read()
        result = await speech_router.process_speech(audio_content, "batch", speech_context)  # Assume batch for uploads
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))

# ===== Analytics and Webhooks =====
@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    handler: InputHandler = Depends(get_input_handler)
):
    try:
        meta = json.loads(metadata) if metadata else {}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            file_path = temp_file.name
        file_id = handler.upload_historical_file(file_path, meta)  # Assume handler handles content
        os.unlink(file_path)
        return {"file_id": file_id, "message": "Uploaded"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/webhooks/twilio/call-status")
async def twilio_webhook(request: Request, handler: InputHandler = Depends(get_input_handler)):
    try:
        form = await request.form()
        data = dict(form)
        result = handler.process_twilio_webhook(data)
        if learning_hook_enabled and data.get('CallStatus') == 'completed':
            conv_data = {}  # Fetch from mongo if needed
            payload = CallEndPayload(**data)
            await handle_call_completion(payload, conv_data.get('entities'), conv_data.get('escalation'))
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/call-patterns")
async def get_call_patterns(mongo: AsyncMongoConnector = Depends(get_mongo)):
    try:
        collection = await mongo.get_collection("logs")
        logs = await collection.find({}).to_list(length=100)
        memory_manager = MemoryManager() # Assumes MemoryManager can extract patterns
        patterns = {}
        for log in logs:
            pattern_data = await memory_manager.analyze_conversation(log.get("bot_response", ""), [log])
            for ptype, pdetails in pattern_data.items():
                if ptype not in patterns:
                    patterns[ptype] = []
                patterns[ptype].append({
                    "type": ptype,
                    "phrase": pdetails.get("phrase", ""),
                    "context": pdetails.get("context", ""),
                    "outcome": "success" if log.get("bot_response", "").lower().includes("approved") else "pending",
                    "source_id": log["call_sid"],
                })
        return {
            "patterns": patterns,
            "total_count": sum(len(p) for p in patterns.values()),
            "types": list(patterns.keys()),
        }
    except Exception as e:
        logger.error(f"Failed to fetch call patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Queue and Cleanup =====
@app.get("/queue/status")
async def queue_status(telephony: EnhancedTelephonyService = Depends(get_telephony_service)):
    if not queue_processing_enabled:
        return {"enabled": False}
    try:
        status = get_queue_status()
        health = queue_health_check.delay().get(timeout=5)
        active_call = next(iter(telephony.active_calls), None)  # Get first active call SID
        return {"status": status, "health": health, "call_sid": active_call}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/analytics/cleanup")
async def cleanup(days_old: int = 30, handler: InputHandler = Depends(get_input_handler)):
    count = handler.cleanup_old_files(days_old)
    return {"cleaned": count}

# ===== Error Handling =====
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logger.error(f"Error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)