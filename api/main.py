"""
Enhanced main.py with Phase 4 S2S pipeline integration and WebSocket support.
Maintains backward compatibility while adding real-time speech-to-speech capabilities.
"""

import sys
import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv
import websockets

# Import Phase 4 components
from .s2s_pipeline import S2SPipeline, create_s2s_pipeline, CallState
from .telephony import EnhancedTelephonyService, get_telephony_service
from .speech import SpeechProcessingRouter, get_speech_router, ProcessingMode
from .conversation import EnhancedConversationManager, get_conversation_manager, ConversationMode

# Import analytics components
from call_analytics.input_sources.historical_uploader import HistoricalUploader
from call_analytics.database.mongo_connector import AsyncMongoConnector

# Phase 5: Post-call processing and queues
from .hooks.learning_hook import (
    handle_call_completion, 
    CallEndPayload, 
    get_conversation_data
)

from shared.config import config
from shared.logging import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

learning_hook_enabled: bool = False
twilio_processor: Optional[TwilioProcessor] = None
queue_processing_enabled: bool = False

# Initialize FastAPI app
app = FastAPI(
    title="Alfons Prior Authorization Bot - Enhanced",
    version="2.0.0",
    description="Healthcare prior authorization bot with real-time S2S capabilities"
)

# Global service instances
telephony_service: Optional[EnhancedTelephonyService] = None
speech_router: Optional[SpeechProcessingRouter] = None
conversation_manager: Optional[EnhancedConversationManager] = None
s2s_pipeline: Optional[S2SPipeline] = None

# Analytics components
analytics_uploader: Optional[HistoricalUploader] = None
mongo_connector: Optional[AsyncMongoConnector] = None
conversation_analyzer: Optional[ConversationAnalyzer] = None
success_predictor: Optional[SuccessPredictor] = None
pattern_extractor: Optional[PatternExtractor] = None

# Active WebSocket connections for Media Streams
active_websockets: Dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize all services and validate configuration on startup."""
    global telephony_service, speech_router, conversation_manager, s2s_pipeline
    global analytics_uploader, mongo_connector, conversation_analyzer, success_predictor, pattern_extractor
    global twilio_processor, learning_hook_enabled, queue_processing_enabled
    
    logger.info("Starting Alfons Enhanced API...")
    
    # Validate required environment variables
    required_vars = [
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER",
        "BASE_URL", "OPENAI_API_KEY", "REDIS_URL", "MONGODB_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    try:
        # Initialize core services
        telephony_service = EnhancedTelephonyService(enable_streaming=True)
        speech_router = get_speech_router()
        conversation_manager = get_conversation_manager()
        s2s_pipeline = create_s2s_pipeline(enable_rag=True)
        
        # Initialize analytics components
        try:
            analytics_uploader = HistoricalUploader()
            mongo_connector = AsyncMongoConnector()
            await mongo_connector.connect()
            conversation_analyzer = ConversationAnalyzer()
            success_predictor = SuccessPredictor()
            pattern_extractor = PatternExtractor()
            logger.info("Analytics components initialized")
        except Exception as e:
            logger.warning(f"Analytics components failed to initialize: {e}")
        
        # Phase 5: Initialize post-call processing components
        try:
            # Initialize Twilio processor for webhook handling
            twilio_processor = TwilioProcessor()
            
            # Check if Celery/Redis is available for queue processing
            try:
                queue_status = get_queue_status()
                queue_processing_enabled = True
                learning_hook_enabled = True
                logger.info("Phase 5 post-call processing and queues initialized successfully")
            except Exception as queue_e:
                logger.warning(f"Queue processing unavailable: {queue_e}")
                queue_processing_enabled = False
                learning_hook_enabled = False
                
        except Exception as e:
            logger.warning(f"Phase 5 components failed to initialize: {e}")
            twilio_processor = None
        
        # Create static directory
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Perform health checks
        health_checks = await asyncio.gather(
            telephony_service.get_service_health(),
            speech_router.health_check(),
            conversation_manager.health_check(),
            return_exceptions=True
        )
        
        for i, check in enumerate(health_checks):
            service_name = ["telephony", "speech", "conversation"][i]
            if isinstance(check, Exception):
                logger.warning(f"{service_name} service health check failed: {check}")
            elif check.get("status") != "healthy":
                logger.warning(f"{service_name} service status: {check.get('status')}")
        
        logger.info("Alfons Enhanced API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Enhanced root endpoint with Phase 4 capabilities."""
    return {
        "status": "Alfons Enhanced API is running",
        "version": "2.0.0",
        "capabilities": {
            "real_time_s2s": True,
            "batch_processing": True,
            "speech_analysis": True,
            "conversation_ai": True,
            "telephony_integration": True
        },
        "endpoints": {
            "traditional": {
                "trigger_call": "/trigger-call",
                "voice_webhook": "/voice",
                "logs": "/logs",
                "patients": "/patients"
            },
            "s2s_enhanced": {
                "trigger_s2s_call": "/s2s/trigger-call",
                "media_stream": "/ws/media-stream",
                "conversation": "/conversation",
                "speech_processing": "/speech/process"
            },
            "analytics": {
                "upload_audio": "/upload-audio",
                "analytics_data": "/analytics-data",
                "call_patterns": "/call-patterns"
            },
            "phase5_processing": {
                "call_status_webhook": "/webhooks/twilio/call-status",
                "manual_call_end": "/webhooks/twilio/call-end", 
                "call_analytics": "/analytics/call/{call_sid}",
                "processing_stats": "/analytics/processing-stats",
                "retry_processing": "/analytics/retry-processing/{call_sid}",
                "batch_upload": "/analytics/upload-batch",
                "queue_status": "/queue/status",
                "cleanup": "/analytics/cleanup"
            }
        }
    }


@app.get("/health")
async def enhanced_health_check():
    """Comprehensive health check for all services."""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    try:
        # Check core services
        if telephony_service:
            health_data["services"]["telephony"] = await telephony_service.get_service_health()
        
        if speech_router:
            health_data["services"]["speech"] = await speech_router.health_check()
        
        if conversation_manager:
            health_data["services"]["conversation"] = await conversation_manager.health_check()
        
        # Check environment
        health_data["services"]["environment"] = {
            "status": "healthy",
            "base_url": config.BASE_URL,
            "static_dir_exists": os.path.exists("static"),
            "redis_configured": bool(config.REDIS_URL),
            "mongodb_configured": bool(config.MONGODB_URL)
        }
        
        # Phase 5: Check post-call processing services
        if twilio_processor:
            try:
                health_data["services"]["twilio_processor"] = {
                    "status": "healthy",
                    "processing_enabled": True
                }
            except Exception as e:
                health_data["services"]["twilio_processor"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        health_data["services"]["learning_hook"] = {
            "status": "healthy" if learning_hook_enabled else "disabled",
            "enabled": learning_hook_enabled
        }
        
        health_data["services"]["queue_processing"] = {
            "status": "healthy" if queue_processing_enabled else "disabled", 
            "enabled": queue_processing_enabled
        }
        
        # Determine overall status
        service_statuses = [svc.get("status", "unknown") for svc in health_data["services"].values()]
        if "unhealthy" in service_statuses:
            health_data["status"] = "unhealthy"
        elif "degraded" in service_statuses:
            health_data["status"] = "degraded"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_data["status"] = "unhealthy"
        health_data["error"] = str(e)
    
    return health_data


# ============================================================================
# ENHANCED S2S ENDPOINTS
# ============================================================================

@app.post("/s2s/trigger-call")
async def trigger_s2s_call(
    phone_number: str = Form(...),
    use_streaming: bool = Form(True),
    context: Optional[str] = Form(None)
):
    """Trigger enhanced S2S call with real-time capabilities."""
    logger.info(f"Triggering S2S call to: {phone_number}")
    
    try:
        # Parse context if provided
        call_context = {}
        if context:
            try:
                call_context = json.loads(context)
            except json.JSONDecodeError:
                logger.warning("Invalid context JSON, using empty context")
        
        # Add S2S-specific context
        call_context.update({
            "s2s_enabled": True,
            "streaming": use_streaming,
            "call_type": "prior_authorization",
            "initiated_by": "api"
        })
        
        # Trigger call through enhanced telephony service
        call_sid = telephony_service.make_call(
            phone_number=phone_number,
            use_streaming=use_streaming,
            call_context=call_context
        )
        
        return {
            "status": "success",
            "call_sid": call_sid,
            "phone_number": phone_number,
            "streaming_enabled": use_streaming,
            "capabilities": {
                "real_time_response": use_streaming,
                "conversation_ai": True,
                "rag_enhanced": True
            }
        }
        
    except Exception as e:
        logger.error(f"S2S call trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger S2S call: {str(e)}")


@app.websocket("/ws/media-stream")
async def media_stream_websocket(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams - handles real-time audio."""
    await websocket.accept()
    stream_id = None
    
    try:
        logger.info("Media Stream WebSocket connection established")
        
        # Use telephony service's stream handler
        if telephony_service and telephony_service.stream_handler:
            await telephony_service.stream_handler.handle_stream_connection(
                websocket, websocket.url.path
            )
        else:
            logger.error("Stream handler not available")
            await websocket.close(code=1011, reason="Stream handler unavailable")
            
    except WebSocketDisconnect:
        logger.info(f"Media Stream WebSocket disconnected: {stream_id}")
    except Exception as e:
        logger.error(f"Media Stream WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@app.post("/conversation/process")
async def process_conversation_message(
    conversation_id: str = Form(...),
    message: str = Form(...),
    mode: str = Form("text"),
    call_sid: Optional[str] = Form(None)
):
    """Process conversation message with enhanced state management."""
    try:
        # Convert mode string to enum
        conv_mode = ConversationMode(mode)
        
        # Process message through conversation manager
        response, extracted_data = await conversation_manager.process_message(
            conversation_id=conversation_id,
            message=message,
            mode=conv_mode,
            call_sid=call_sid
        )
        
        return {
            "status": "success",
            "response": response,
            "extracted_data": extracted_data,
            "conversation_id": conversation_id,
            "processing_mode": mode
        }
        
    except Exception as e:
        logger.error(f"Conversation processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/conversation/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    """Get comprehensive conversation summary."""
    try:
        summary = await conversation_manager.get_conversation_summary(conversation_id)
        return summary
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech/process")
async def process_speech(
    audio: UploadFile = File(...),
    mode: str = Form("auto"),
    context: Optional[str] = Form(None)
):
    """Process speech through unified speech router."""
    try:
        # Parse context
        speech_context = {}
        if context:
            try:
                speech_context = json.loads(context)
            except json.JSONDecodeError:
                pass
        
        # Convert mode string to enum
        processing_mode = ProcessingMode(mode)
        
        # Read audio file
        audio_content = await audio.read()
        
        # Process through speech router
        result = await speech_router.process_speech(
            audio_input=audio_content,
            mode=processing_mode,
            context=speech_context
        )
        
        return {
            "status": "success",
            "result": result,
            "processing_mode": mode,
            "filename": audio.filename
        }
        
    except Exception as e:
        logger.error(f"Speech processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRADITIONAL TELEPHONY ENDPOINTS (BACKWARD COMPATIBILITY)
# ============================================================================

@app.post("/trigger-call")
async def trigger_traditional_call(phone_number: str = Form(...)):
    """Traditional call trigger (backward compatibility)."""
    logger.info(f"Triggering traditional call to: {phone_number}")
    
    try:
        # Use traditional webhook-based calling
        call_sid = telephony_service.make_call(
            phone_number=phone_number,
            use_streaming=False  # Traditional mode
        )
        
        return {
            "status": "success",
            "call_sid": call_sid,
            "phone_number": phone_number,
            "mode": "traditional"
        }
        
    except Exception as e:
        logger.error(f"Traditional call trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice")
async def voice_get():
    """Voice webhook GET endpoint for verification."""
    return {
        "status": "Voice webhook is accessible",
        "message": "This endpoint handles Twilio voice webhooks via POST",
        "enhanced_features": {
            "s2s_streaming": True,
            "conversation_ai": True,
            "rag_integration": True
        }
    }


@app.post("/voice")
async def voice_webhook(request: Request):
    """Enhanced voice webhook supporting both traditional and streaming modes."""
    logger.info("ENHANCED VOICE WEBHOOK CALLED")
    
    try:
        form = await request.form()
        form_dict = dict(form)
        logger.info(f"Voice webhook form data: {form_dict}")
        
        call_sid = form.get("CallSid")
        audio_url = form.get("RecordingUrl")
        call_status = form.get("CallStatus")
        
        # Check if this call should use streaming
        call_info = telephony_service.get_call_status(call_sid) if call_sid else None
        use_streaming = call_info.get("use_streaming", False) if call_info else False
        
        # Route to appropriate handler
        if use_streaming:
            # Generate TwiML for streaming-enabled calls
            twiml_response = telephony_service.generate_streaming_twiml(call_sid)
        else:
            # Handle traditional webhook processing
            if audio_url:
                # Process recorded audio through speech router
                result = await speech_router.process_speech(
                    audio_input=audio_url,
                    mode=ProcessingMode.BATCH,
                    context={"call_sid": call_sid, "legacy_mode": True}
                )
                
                transcript = result.get("transcript", "")
                
                # Process through conversation manager
                conversation_id = f"call_{call_sid}"
                response_text, extracted_data = await conversation_manager.process_message(
                    conversation_id=conversation_id,
                    message=transcript,
                    mode=ConversationMode.TEXT,
                    call_sid=call_sid
                )
                
                # Generate TwiML response
                twiml_response = telephony_service.generate_webhook_twiml(call_sid, audio_url)
                
                # Log conversation
                await log_conversation(call_sid, transcript, response_text, extracted_data)
            else:
                # Initial call setup
                twiml_response = telephony_service.generate_webhook_twiml(call_sid)
        
        return Response(content=twiml_response, media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Voice webhook error: {e}")
        
        # Emergency fallback TwiML
        emergency_twiml = VoiceResponse()
        emergency_twiml.say("I apologize, but I'm experiencing technical difficulties. Please try calling again in a few minutes.")
        return Response(content=str(emergency_twiml), media_type="text/xml")


@app.post("/voice/streaming")
async def voice_streaming_webhook(request: Request):
    """Voice webhook specifically for streaming-enabled calls."""
    logger.info("STREAMING VOICE WEBHOOK CALLED")
    
    try:
        form = await request.form()
        call_sid = form.get("CallSid")
        
        if call_sid:
            # Handle call status updates for streaming calls
            await telephony_service.handle_call_status_update(
                call_sid, form.get("CallStatus", "unknown"), **dict(form)
            )
        
        # Generate streaming TwiML
        twiml_response = telephony_service.generate_streaming_twiml(call_sid)
        return Response(content=twiml_response, media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Streaming voice webhook error: {e}")
        emergency_twiml = VoiceResponse()
        emergency_twiml.say("I'm sorry, there was an issue with the streaming connection.")
        return Response(content=str(emergency_twiml), media_type="text/xml")


# ============================================================================
# DATA AND ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/logs")
async def fetch_logs():
    """Fetch conversation logs (backward compatibility)."""
    try:
        logs = await get_logs()
        logger.info(f"Fetched {len(logs)} logs")
        return logs
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients")
async def fetch_patients():
    """Fetch all patients requiring prior authorization."""
    try:
        patients = await get_patients()
        logger.info(f"Fetched {len(patients)} patients")
        return patients
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
async def get_patient_details(patient_id: int):
    """Fetch details for a specific patient by ID."""
    try:
        patient = await get_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/patients/{patient_id}/auth-status")
async def update_auth_status(patient_id: int, status: str = Form(...)):
    """Update prior authorization status for a patient."""
    try:
        valid_statuses = ["Pending", "Approved", "Denied", "Under Review"]
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        updated_patient = await update_patient_auth_status(patient_id, status)
        logger.info(f"Updated patient {patient_id} auth status to {status}")
        return {"status": "success", "patient": updated_patient}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating patient {patient_id} auth status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-audio")
async def upload_audio_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Upload historical call audio files for analysis."""
    if not analytics_uploader:
        raise HTTPException(status_code=503, detail="Analytics service unavailable")
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Only MP3, WAV, or M4A files allowed")
    
    try:
        # Save temp file
        temp_path = f"./temp_uploads/{file.filename}"
        os.makedirs("./temp_uploads", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse metadata
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON")
        
        # Add processing timestamp
        meta_dict["uploaded_at"] = datetime.utcnow().isoformat()
        meta_dict["file_size"] = len(content)
        
        # Upload and process
        file_id = analytics_uploader.upload_file(temp_path, meta_dict)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info(f"Audio file uploaded successfully: {file_id}")
        return {
            "success": True,
            "file_id": file_id,
            "message": "Audio uploaded and processing started",
            "metadata": meta_dict
        }
        
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics-data")
async def get_analytics_data(limit: int = 20):
    """Get analytics data from processed calls."""
    if not mongo_connector:
        raise HTTPException(status_code=503, detail="Analytics service unavailable")
    
    try:
        # Get recent uploads with processing status
        uploads = await mongo_connector.find_documents("uploads", {}, limit=limit)
        analytics = await mongo_connector.find_documents("analytics", {}, limit=limit)
        
        # Calculate summary statistics
        total_uploads = len(uploads)
        processed_count = len([u for u in uploads if u.get("processed", False)])
        success_count = len([a for a in analytics if a.get("outcome") == "success"]) if analytics else 0
        success_rate = success_count / len(analytics) if analytics else 0.0
        
        return {
            "uploads": uploads,
            "analytics": analytics,
            "summary": {
                "total_uploads": total_uploads,
                "processed_count": processed_count,
                "success_rate": success_rate,
                "processing_rate": processed_count / total_uploads if total_uploads > 0 else 0.0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching analytics data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/call-patterns")
async def get_call_patterns(pattern_type: Optional[str] = None, limit: int = 50):
    """Get extracted patterns from analyzed calls."""
    if not mongo_connector:
        raise HTTPException(status_code=503, detail="Analytics service unavailable")
    
    try:
        query = {}
        if pattern_type:
            query = {"patterns.type": pattern_type}
        
        patterns_data = await mongo_connector.find_documents("training_data", query, limit=limit)
        
        # Extract and format patterns
        all_patterns = []
        for item in patterns_data:
            patterns = item.get("patterns", [])
            for pattern in patterns:
                pattern["source_id"] = str(item.get("_id", ""))
                pattern["created_at"] = item.get("created_at", "")
                all_patterns.append(pattern)
        
        # Group by type
        grouped_patterns = {}
        for pattern in all_patterns:
            ptype = pattern.get("type", "unknown")
            if ptype not in grouped_patterns:
                grouped_patterns[ptype] = []
            grouped_patterns[ptype].append(pattern)
        
        return {
            "patterns": grouped_patterns,
            "total_count": len(all_patterns),
            "types": list(grouped_patterns.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching call patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PHASE 5: POST-CALL PROCESSING & QUEUE ENDPOINTS
# ============================================================================

@app.post("/webhooks/twilio/call-status")
async def handle_twilio_call_status_webhook(request: Request):
    """
    Enhanced Twilio webhook handler for call status updates.
    Triggers post-call processing and learning pipeline.
    """
    if not twilio_processor:
        logger.warning("Twilio processor not available")
        return {"status": "processor_unavailable"}
    
    try:
        form = await request.form()
        webhook_data = dict(form)
        
        logger.info(f"Received Twilio call status webhook: {webhook_data.get('CallSid')} - {webhook_data.get('CallStatus')}")
        
        # Process through enhanced Twilio processor
        result = twilio_processor.process_call_webhook(webhook_data)
        
        # If call completed and learning hook is enabled, trigger learning pipeline
        if (webhook_data.get('CallStatus') == 'completed' and 
            learning_hook_enabled and 
            webhook_data.get('CallSid')):
            
            try:
                # Get conversation data from database
                conversation_data = await get_conversation_data(webhook_data.get('CallSid'))
                
                # Prepare call end payload
                call_payload = CallEndPayload(**webhook_data)
                
                # Trigger learning hook
                learning_result = await handle_call_completion(
                    payload=call_payload,
                    conversation_entities=conversation_data.get('entities') if conversation_data else None,
                    escalation_data=conversation_data.get('escalation') if conversation_data else None
                )
                
                logger.info(f"Learning hook result: {learning_result}")
                result['learning_hook'] = learning_result
                
            except Exception as learning_e:
                logger.error(f"Learning hook failed for call {webhook_data.get('CallSid')}: {learning_e}")
                result['learning_hook'] = {'status': 'error', 'error': str(learning_e)}
        
        return result
        
    except Exception as e:
        logger.error(f"Twilio webhook processing failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/webhooks/twilio/call-end")
async def manual_call_end_trigger(
    call_sid: str = Form(...),
    call_status: str = Form("completed"),
    call_duration: Optional[str] = Form(None),
    recording_url: Optional[str] = Form(None)
):
    """
    Manual trigger for call end processing (for testing or manual intervention).
    """
    if not learning_hook_enabled:
        raise HTTPException(status_code=503, detail="Learning hook not available")
    
    try:
        # Create call end payload
        payload_data = {
            'CallSid': call_sid,
            'CallStatus': call_status,
            'CallDuration': call_duration,
            'RecordingUrl': recording_url
        }
        
        call_payload = CallEndPayload(**payload_data)
        
        # Get conversation data
        conversation_data = await get_conversation_data(call_sid)
        
        # Trigger learning hook
        result = await handle_call_completion(
            payload=call_payload,
            conversation_entities=conversation_data.get('entities') if conversation_data else None,
            escalation_data=conversation_data.get('escalation') if conversation_data else None
        )
        
        return {
            "status": "success",
            "call_sid": call_sid,
            "learning_result": result,
            "conversation_data_found": conversation_data is not None
        }
        
    except Exception as e:
        logger.error(f"Manual call end trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/call/{call_sid}")
async def get_call_analytics_detailed(call_sid: str):
    """
    Get comprehensive analytics for a specific call including queue processing status.
    """
    if not twilio_processor:
        raise HTTPException(status_code=503, detail="Twilio processor not available")
    
    try:
        # Get analytics from Twilio processor
        analytics = twilio_processor.get_call_analytics(call_sid)
        
        # Get conversation data from database
        conversation_data = await get_conversation_data(call_sid)
        
        # Combine all data
        comprehensive_analytics = {
            "call_sid": call_sid,
            "twilio_analytics": analytics,
            "conversation_data": conversation_data,
            "queue_processing_enabled": queue_processing_enabled,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return comprehensive_analytics
        
    except Exception as e:
        logger.error(f"Error getting call analytics for {call_sid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/processing-stats")
async def get_processing_statistics(hours_back: int = 24):
    """
    Get comprehensive processing statistics including queue metrics.
    """
    try:
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "hours_back": hours_back,
            "services": {
                "learning_hook": learning_hook_enabled,
                "queue_processing": queue_processing_enabled,
                "twilio_processor": twilio_processor is not None
            }
        }
        
        # Get Twilio processor stats
        if twilio_processor:
            stats["twilio_stats"] = twilio_processor.get_processing_stats(hours_back)
        
        # Get queue stats
        if queue_processing_enabled:
            try:
                stats["queue_stats"] = get_queue_status()
            except Exception as e:
                stats["queue_stats"] = {"error": str(e)}
        
        # Get analytics uploader stats
        if analytics_uploader:
            stats["uploader_stats"] = analytics_uploader.get_processing_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/retry-processing/{call_sid}")
async def retry_call_processing(call_sid: str):
    """
    Retry processing for a failed call.
    """
    if not twilio_processor:
        raise HTTPException(status_code=503, detail="Twilio processor not available")
    
    try:
        result = twilio_processor.retry_failed_processing(call_sid)
        return result
        
    except Exception as e:
        logger.error(f"Error retrying processing for {call_sid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/upload-batch")
async def upload_batch_files(
    files: list[UploadFile] = File(...),
    batch_metadata: Optional[str] = Form(None)
):
    """
    Upload multiple audio files as a batch for historical analysis.
    """
    if not analytics_uploader:
        raise HTTPException(status_code=503, detail="Analytics uploader not available")
    
    try:
        # Parse batch metadata
        batch_meta = {}
        if batch_metadata:
            try:
                batch_meta = json.loads(batch_metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid batch metadata JSON")
        
        # Save temporary files
        temp_paths = []
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        for file in files:
            if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                continue
                
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(temp_path)
        
        # Upload batch
        batch_result = analytics_uploader.upload_batch(temp_paths, batch_meta)
        
        # Clean up temp files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return {
            "status": "success",
            "batch_result": batch_result,
            "files_processed": len(temp_paths)
        }
        
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        # Clean up temp files on error
        if 'temp_paths' in locals():
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/status")
async def get_queue_processing_status():
    """
    Get current status of the processing queue.
    """
    if not queue_processing_enabled:
        return {
            "queue_enabled": False,
            "message": "Queue processing not available"
        }
    
    try:
        # Get queue status
        queue_status = get_queue_status()
        
        # Get queue health check
        health_result = queue_health_check.delay()
        health_status = health_result.get(timeout=5)  # 5 second timeout
        
        return {
            "queue_enabled": True,
            "status": queue_status,
            "health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {
            "queue_enabled": True,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.delete("/analytics/cleanup")
async def cleanup_old_files(
    days_old: int = 30,
    cleanup_recordings: bool = True,
    cleanup_uploads: bool = True
):
    """
    Clean up old files to manage disk space.
    """
    results = {
        "cleanup_started_at": datetime.utcnow().isoformat(),
        "days_old": days_old,
        "results": {}
    }
    
    try:
        # Clean up old uploads
        if cleanup_uploads and analytics_uploader:
            upload_cleanup_count = analytics_uploader.cleanup_old_files(days_old)
            results["results"]["uploads_cleaned"] = upload_cleanup_count
        
        # Clean up old recordings
        if cleanup_recordings and twilio_processor:
            recording_cleanup_count = twilio_processor.cleanup_old_recordings(days_old)
            results["results"]["recordings_cleaned"] = recording_cleanup_count
        
        results["status"] = "success"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        return results
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        results["status"] = "error"
        results["error"] = str(e)
        return results


# ============================================================================
# MONITORING AND STATUS ENDPOINTS
# ============================================================================

@app.get("/status/calls")
async def get_active_calls():
    """Get status of active calls."""
    try:
        active_calls = telephony_service.get_active_calls() if telephony_service else []
        return {
            "active_calls": active_calls,
            "count": len(active_calls),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/speech")
async def get_speech_processing_stats():
    """Get speech processing statistics."""
    try:
        stats = speech_router.get_processing_stats() if speech_router else {}
        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting speech stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/websockets")
async def get_websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": len(active_websockets),
        "connection_ids": list(active_websockets.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with enhanced logging."""
    logger.error(f"Global exception on {request.url}: {str(exc)}")
    logger.error(f"Exception type: {type(exc).__name__}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )