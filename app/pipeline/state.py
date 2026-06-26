from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    user_message: str
    conversation_history: List[dict]
    intent: Optional[str]
    entities: Optional[dict]
    retrieved_schemes: Optional[List[dict]]
    final_response: Optional[str]
    needs_clarification: Optional[bool]
