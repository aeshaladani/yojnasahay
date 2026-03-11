from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile, os, shutil
from pipeline import chat

app = FastAPI(title="YojnaSahay API")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper model (loaded once at startup)
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
    return {"status": "YojnaSahay API is running", "whisper": "ready"}

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

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (webm/wav/mp3),
    transcribes it using faster-whisper,
    returns the text and detected language.
    """
    # Save uploaded audio to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        model = get_whisper()
        # language=None lets Whisper auto-detect Hindi or English
        # First pass: detect language
        _, info = model.transcribe(tmp_path, beam_size=2, without_timestamps=True)
        detected_lang = info.language or "en"

        # Second pass: transcribe with correct language forced
        # If Hindi detected, force Devanagari script output
        if detected_lang == "hi":
            segments, info = model.transcribe(
                tmp_path,
                language="hi",
                beam_size=5,
                task="transcribe",
                without_timestamps=True,
                initial_prompt="यह एक हिंदी वाक्य है।"  # primes Whisper to output Devanagari
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
        language = detected_lang
        return TranscribeResponse(text=text, language=language)
    finally:
        os.unlink(tmp_path)  # clean up temp file

@app.post("/reset")
def reset():
    return {"message": "Send conversation_history: [] to start fresh."}