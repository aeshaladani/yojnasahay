import json
from langchain_core.messages import HumanMessage
from app.pipeline.state import AgentState
from app.services.llm import llm

# Node 1: Intent Detection
def detect_intent(state: AgentState) -> AgentState:
    print("\n[Node 1] Detecting intent...")

    history_text = ""
    for msg in state["conversation_history"][-4:]:
        history_text += f"{msg.get('role')}: {msg.get('content')}\n"

    prompt = f"""You are an assistant that helps Indian citizens find government schemes.
Analyze the user message and classify the intent into ONE of these:
- find_scheme: user wants to find NEW government schemes
- get_details: user wants MORE details about a scheme already mentioned in conversation
- eligibility_check: user wants to check eligibility for a scheme
- general: general question or greeting
- translate: user wants to translate or repeat the last response in a different language (e.g. "translate to hindi", "hindi mein batao", "hindi me do", "translate this", "anuvad karo", "english mein batao")
Conversation history:
{history_text}

Current message: "{state['user_message']}"

If the user is asking about a scheme that was already mentioned above (e.g. "mukhyamantri jankalyan", "tell me more", "benefits", "documents", "how to apply", "eligibility"), classify as get_details.
If the user is asking for a new search or new type of scheme, classify as find_scheme.

Respond with ONLY a JSON object like this:
{{"intent": "find_scheme", "confidence": "high"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        intent = result.get("intent", "find_scheme")
    except:
        intent = "find_scheme"

    print(f"  Intent: {intent}")
    return {**state, "intent": intent}
