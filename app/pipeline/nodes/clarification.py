from app.pipeline.state import AgentState

# Node 3: Check if clarification needed (BEFORE retrieval to avoid unnecessary Qdrant calls)
def check_clarification(state: AgentState) -> AgentState:
    entities = state.get("entities") or {}
    history = state.get("conversation_history") or []

    is_first_message = len(history) <= 1
    has_no_info = not any([
        entities.get("state"),
        entities.get("occupation"),
        entities.get("category"),
        entities.get("caste")
    ])

    needs_clarification = is_first_message and has_no_info
    return {**state, "needs_clarification": needs_clarification}
