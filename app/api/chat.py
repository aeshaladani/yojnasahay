from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse, Message
from app.pipeline.graph import chat

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
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
