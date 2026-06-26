import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from app.models.schemas import TranscribeResponse
from app.services.whisper import get_whisper, transcribe_with_groq

router = APIRouter()

@router.options("/transcribe")
async def transcribe_options(request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (webm/wav/mp3).
    Primary: Groq Whisper API (no RAM cost).
    Fallback: local faster-whisper (only if Groq fails).
    """
    suffix = os.path.splitext(file.filename)[-1] or ".webm"
    audio_bytes = await file.read()

    # PRIMARY: Groq Whisper API
    result = await transcribe_with_groq(audio_bytes, file.filename, suffix)
    if result["success"]:
        return TranscribeResponse(text=result["text"], language=result["language"])

    # FALLBACK: local faster-whisper (only if Groq unavailable/failed)
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
