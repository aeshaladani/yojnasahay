import json
from langchain_core.messages import HumanMessage
from app.pipeline.state import AgentState
from app.services.llm import llm

# Node 2: Entity Extraction
def extract_entities(state: AgentState) -> AgentState:
    print("\n[Node 2] Extracting entities...")

    history_text = ""
    for msg in state["conversation_history"][-4:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    prompt = f"""Extract key information from the conversation to find relevant government schemes.

Conversation history:
{history_text}

Current message: "{state['user_message']}"

Extract these fields (use null if not mentioned):
- state: Indian state name (e.g. Gujarat, Bihar, Maharashtra)
- age: numeric age
- income: annual family income in rupees
- gender: male/female/any
- caste: SC/ST/OBC/General/any
- occupation: farmer/student/worker/entrepreneur/any
- category: scheme category like education/agriculture/health/housing/women

Respond ONLY with a JSON object with these exact keys. Use null for any field not mentioned:
{{"state": null, "age": null, "income": null, "gender": "any", "caste": "any", "occupation": "any", "category": null}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        entities = json.loads(raw)
    except:
        entities = {}

    # Merge with existing entities - new values always override old ones
    existing = state.get("entities") or {}
    merged = {**existing}
    for k, v in entities.items():
        if v is not None and v != "null" and v != "any":
            merged[k] = v  # always override, so "income is 10 lakh" updates previous income
        elif v == "any" and k not in existing:
            merged[k] = v  # only set "any" if not already known

    print(f"  Entities: {merged}")
    return {**state, "entities": merged}
