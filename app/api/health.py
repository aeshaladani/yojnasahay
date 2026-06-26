from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {"status": "YojnaSahay API is running", "whisper": "groq-api"}

@router.post("/reset")
def reset():
    return {"message": "Send conversation_history: [] to start fresh."}
