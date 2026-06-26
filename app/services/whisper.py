import os
import httpx

whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model loaded!")
    return whisper_model

async def transcribe_with_groq(audio_bytes: bytes, filename: str, suffix: str) -> dict:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                    files={"file": (filename or f"audio{suffix}", audio_bytes, "audio/webm")},
                    data={"model": "whisper-large-v3", "language": "hi"}
                )
            if response.status_code == 200:
                result = response.json()
                text = result.get("text") or result.get("transcript") or ""
                # Detect language: if Devanagari chars found → Hindi, else English
                language = "hi" if any("\u0900" <= ch <= "\u097F" for ch in text) else "en"
                return {"success": True, "text": text.strip(), "language": language}
            else:
                print(f"Groq API error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Groq API failed, falling back to local Whisper: {e}")
    return {"success": False}
