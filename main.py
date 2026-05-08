from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pipeline import chat
from fastapi import Request
from fastapi.responses import JSONResponse
import tempfile, os, shutil, httpx
app = FastAPI(title="YojnaSahay API")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper model — lazy loaded, only used if Groq API fails
whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model loaded!")
    return whisper_model

# Request / Response Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Message]

class TranscribeResponse(BaseModel):
    text: str
    language: str

# Routes 
@app.get("/")
def root():
    return {"status": "YojnaSahay API is running", "whisper": "groq-api"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]
    response_text = chat(req.message, history)
    updated_history = history + [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": response_text},
    ]
    return ChatResponse(
        response=response_text,
        conversation_history=[Message(**m) for m in updated_history]
    )

@app.options("/transcribe")
async def transcribe_options(request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (webm/wav/mp3).
    Primary: Groq Whisper API (no RAM cost).
    Fallback: local faster-whisper (only if Groq fails).
    """
    suffix = os.path.splitext(file.filename)[-1] or ".webm"
    audio_bytes = await file.read()

    # ── PRIMARY: Groq Whisper API ──────────────────────────────────────────
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                    files={"file": (file.filename or f"audio{suffix}", audio_bytes, "audio/webm")},
                    data={"model": "whisper-large-v3", "language": "hi"}
                )
            if response.status_code == 200:
                result = response.json()
                text = result.get("text") or result.get("transcript") or ""
                # Detect language: if Devanagari chars found → Hindi, else English
                language = "hi" if any("\u0900" <= ch <= "\u097F" for ch in text) else "en"
                return TranscribeResponse(text=text.strip(), language=language)
            else:
                print(f"Groq API error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Groq API failed, falling back to local Whisper: {e}")

    # ── FALLBACK: local faster-whisper (only if Groq unavailable/failed) ──
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = get_whisper()
        # First pass: detect language
        _, info = model.transcribe(tmp_path, beam_size=2, without_timestamps=True)
        detected_lang = info.language or "en"

        # Second pass: transcribe with correct language forced
        if detected_lang == "hi":
            segments, info = model.transcribe(
                tmp_path,
                language="hi",
                beam_size=5,
                task="transcribe",
                without_timestamps=True,
                initial_prompt="यह एक हिंदी वाक्य है।"
            )
        else:
            segments, info = model.transcribe(
                tmp_path,
                language=detected_lang,
                beam_size=5,
                task="transcribe",
                without_timestamps=True
            )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscribeResponse(text=text, language=detected_lang)
    finally:
        os.unlink(tmp_path)

@app.post("/reset")
def reset():
    return {"message": "Send conversation_history: [] to start fresh."}