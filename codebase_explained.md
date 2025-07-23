Alfons Project: AI-Powered Prior Authorization Bot - Comprehensive Documentation and Enhancement Prompt
Introduction and Project Overview
You are Grok, an advanced AI built by xAI, tasked with enhancing the Alfons project—a streamlined, MVP-focused AI agent for healthcare prior authorization calls. Alfons uses real-time speech-to-speech (S2S) capabilities with Twilio integration, OpenAI (or swappable LLMs via providers), and analytics for post-call learning. The codebase emphasizes simplicity: files <500 lines, modularity via FastAPI routers/dependencies, async operations, provider abstractions for easy swaps (LLM/STT/TTS), in-memory/MongoDB state, Celery queues for async tasks, and no backward compatibility.

This document is your complete context for the project. Use it to:

Understand the current architecture.
Implement new features without redundancy (e.g., reuse providers, dependencies, existing classes/methods).
Follow best practices: async-first, dependency injection, logging via shared.logging, config from .env/ shared.config, error handling with FastAPI exceptions, health checks in services.
Avoid: Globals (use Depends), multiple services per process (single provider), complex state (prefer in-memory for MVP).
When adding: Keep files <400 lines, test with snippets, integrate with main.py routers.
Current date: July 20, 2025. Build on this setup to add features like advanced RAG, HIPAA compliance, or UI integration.

Core Goals
Real-time S2S for calls: Audio → STT → LLM/Conversation → TTS → Response.
Batch analytics: Upload/transcribe/analyze/learn from historical calls.
Flexibility: Swap LLMs/STT/TTS via shared/providers.
Maintenance: Simple structure, detailed logging, health endpoints.
Tech Stack
Backend: FastAPI (async API), Celery (queues), Redis (state/cache), MongoDB (persistent data).
AI: OpenAI (default; swappable), Langchain (minimal for messages).
Telephony: Twilio (streams/webhooks).
Audio: FFmpeg (processing), providers for STT/TTS.
Env: .env for keys/URLs.
File Structure
The project is organized as follows (simplified tree):

Best Practices in Codebase
Dependencies: Use FastAPI Depends for services (e.g., telephony: EnhancedTelephonyService = Depends(get_telephony_service)). Avoid globals.
Providers: All AI/audio via shared/providers (e.g., get_llm() returns Langchain-compatible model). Swap by config/env (no code change).
Async: All I/O (DB, API, streams) async with asyncio.
Logging: Centralized via get_logger(__name__); info/debug/error levels.
Error Handling: FastAPI exceptions, try/except with logs, health checks return dicts with status/error.
MVP Focus: In-memory where possible (e.g., sessions); Mongo for persistence; simple rules over complex ML.
Health: Each service has async def health_check(self) -> Dict; aggregated in main.py /health.
Queues: Celery for post-call (process_call task chains transcribe/analyze/update).
Cleanup: Temp files deleted post-use; old files via cron-like methods.
Config: All from .env via shared.config; no hardcodes.
Code Snippet (Dependency Example from main.py):

python

Collapse

Wrap

Run

Copy
from .telephony import get_telephony_service

@app.post("/s2s/trigger-call")
async def trigger_call(telephony: EnhancedTelephonyService = Depends(get_telephony_service)):
    # Use telephony
Key Components with Code Snippets
1. shared/providers (Abstractions for Swaps)
illm_provider.py: Returns LLM instance. Snippet:
python

Collapse

Wrap

Run

Copy
from langchain_openai import ChatOpenAI
from shared.config import config

def get_llm(model: str = "gpt-4", **kwargs):
    return ChatOpenAI(api_key=config.OPENAI_API_KEY, model=model, **kwargs)
Similar for istt/itts (e.g., Deepgram/OpenAI for STT/TTS).
2. call_analytics Files (Analytics/Processing Backend)
analytics_engine.py: Combined analysis (objections, success prediction, timing). Snippet (Core Method):
python

Collapse

Wrap

Run

Copy
def analyze_conversation(self, transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Extracts entities, objections, timing, etc.
    return {"entities": ..., "objections": ..., "timing": ...}
input_handler.py: Merged uploads/Twilio; queues to Celery. Snippet (Upload):
python

Collapse

Wrap

Run

Copy
def upload_historical_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    # Copy, meta, insert DB, queue process_call
learning_pipeline.py: Pattern extraction, script gen, memory, training. Snippet (Memory Store):
python

Collapse

Wrap

Run

Copy
def store_memory(self, memory: Dict[str, Any]):
    mtype = memory["type"]
    self.vector_index.insert_with_embedding(mtype, memory, "text")
mongo_connector.py: Async Mongo ops. Snippet:
python

Collapse

Wrap

Run

Copy
async def insert_document(self, collection: str, doc: Dict[str, Any]) -> str:
    result = await self.db[collection].insert_one(doc)
    return str(result.inserted_id)
task_queue.py: Simple Celery tasks. Snippet (Main Task):
python

Collapse

Wrap

Run

Copy
@app.task
def process_call(call_data: Dict[str, Any]) -> Dict[str, Any]:
    # Chain: transcribe → analyze → update_memory
transcription_pipeline.py: Batch STT/diarization/validation. Snippet:
python

Collapse

Wrap

Run

Copy
async def transcribe_audio(self, audio_path: str) -> str:
    # Enhance, chunk, STT via provider, diarize/validate inline
vector_index.py: Simple vector ops (assume Mongo Atlas or embed).
3. api Files (Core API/Telephony)
conversation.py: State/LLM/extraction. Snippet (Process):
python

Collapse

Wrap

Run

Copy
async def process_message(self, conv_id: str, message: str, ...) -> Tuple[str, Dict]:
    # Extract, generate response, update state
s2s_pipeline.py: Realtime audio processing. Snippet (Chunk Process):
python

Collapse

Wrap

Run

Copy
async def process_audio_chunk(self, call_id: str, audio_data: bytes) -> AsyncIterable[bytes]:
    # STT, converse, TTS yield
speech.py: Router for STT/TTS/batch. Snippet:
python

Collapse

Wrap

Run

Copy
async def process_speech(self, audio_input, mode="auto", ...):
    # Auto route to realtime or batch
telephony.py: Twilio calls/streams. Snippet (WS Handle):
python

Collapse

Wrap

Run

Copy
async def handle_stream_connection(self, websocket, path):
    # Process events, delegate to S2S
main.py: FastAPI app (as generated: deps, endpoints, startup).
4. hooks/learning_hook.py (Post-Call)
Assume simple: Trigger learning on call end. Snippet:
python

Collapse

Wrap

Run

Copy
async def handle_call_completion(payload: CallEndPayload, ...):
    # Use learning_pipeline to process
How the Project is Built
Startup Flow (main.py): Validate env, init services via factories (e.g., create_s2s_pipeline()), mount static, health checks.
Call Flow: /s2s/trigger-call → telephony.make_call → Twilio call → WS /media-stream → telephony.handle → s2s.process_chunk (STT/LLM/TTS via providers/conversation).
Batch/Analytics: /upload-audio → input_handler.upload → queue process_call → transcribe (pipeline) → analyze (engine) → learn (pipeline).
State/Memory: In-memory with Mongo backup; simple enums.
Extensions: Add to existing (e.g., RAG to conversation._generate_response via LLM chain).
Prompt for Enhancement
To add features:

Reuse: Providers for AI, deps for services, async for I/O.
Avoid: Duplicates (check if method exists, e.g., in input_handler for uploads).
Test: Add snippets, health checks.
New Feature Example: Add RAG—integrate retriever in conversation.py _generate_response.
Enhance by implementing [new feature], building on this.