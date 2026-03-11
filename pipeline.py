import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

load_dotenv()

# Init
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)


qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "schemes"

# State Definition 
class AgentState(TypedDict):
    user_message: str
    conversation_history: List[dict]
    intent: Optional[str]
    entities: Optional[dict]
    retrieved_schemes: Optional[List[dict]]
    final_response: Optional[str]
    needs_clarification: Optional[bool]

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

    # Merge with existing entities from previous turns
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

# Node 3: RAG Retrieval
def retrieve_schemes(state: AgentState) -> AgentState:
    print("\n[Node 3] Retrieving schemes from Qdrant...")

    entities = state.get("entities") or {}

    # Build search query
    query_parts = [state["user_message"]]
    if entities.get("occupation"): query_parts.append(entities["occupation"])
    if entities.get("category"):   query_parts.append(entities["category"])
    if entities.get("state"):      query_parts.append(entities["state"])
    if entities.get("caste"):      query_parts.append(entities["caste"])
    search_query = " ".join(query_parts)

    query_vector = embedder.encode(search_query).tolist()

    # Build state filter
    search_filter = None
    if entities.get("state"):
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="eligibility_raw",
                    match=MatchText(text=entities["state"])
                )
            ]
        )

    # Try filtered search first, fallback to unfiltered
    results = []
    if search_filter:
        try:
            results = qdrant.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=15,
                query_filter=search_filter
            ).points
            print(f"  Found {len(results)} schemes with state filter")
        except:
            pass

    if not results:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=15
        ).points
        print(f"  Found {len(results)} schemes without filter (fallback)")

    schemes = [r.payload for r in results]

    # Caste Filter (Python-level post-processing)
    user_caste = (entities.get("caste") or "any").lower()

    caste_keywords = {
        "sc": ["scheduled caste", "sc students", "sc category", "sc/st"],
        "st": ["scheduled tribe", "st students", "st category"],
        "obc": ["other backward", "obc students", "obc category"],
        "general": ["general category", "general merit", "non-reserved"],
    }

    def is_exclusive_for_other_caste(scheme, user_caste):
        if user_caste == "any":
            return False
        text = (
            scheme.get("scheme_name", "") + " " +
            scheme.get("eligibility_raw", "")
        ).lower()
        for caste, keywords in caste_keywords.items():
            if caste == user_caste:
                continue
            if any(kw in text for kw in keywords):
                if user_caste == "general" and caste in ["sc", "st", "obc"]:
                    return True
                if user_caste == "sc" and caste in ["st", "obc"]:
                    return True
                if user_caste == "st" and caste in ["sc", "obc"]:
                    return True
                if user_caste == "obc" and caste in ["sc", "st"]:
                    return True
        return False

    if user_caste != "any":
        before = len(schemes)
        schemes = [s for s in schemes if not is_exclusive_for_other_caste(s, user_caste)]
        after = len(schemes)
        print(f"  Caste filter: {before} → {after} schemes after filtering for {user_caste.upper()}")

    # Gender Filter (Python-level post-processing)
    user_gender = (entities.get("gender") or "any").lower()

    girl_keywords = [
        "girl child", "girls only", "women only", "only women", "only girls",
        "female students", "for girls", "for women", "girl student", "women student",
        "kanyashree", "ladli", "beti", "mahila", "stree", "balika",
        "single girl", "daughters", "women empowerment"
    ]

    boy_keywords = [
        "only for boys", "only men", "men only", "male students only", "for boys only"
    ]

    def is_exclusive_for_other_gender(scheme, user_gender):
        if user_gender == "any":
            return False
        text = (
            scheme.get("scheme_name", "") + " " +
            scheme.get("eligibility_raw", "")
        ).lower()
        if user_gender == "male":
            if any(kw in text for kw in girl_keywords):
                return True
        elif user_gender == "female":
            if any(kw in text for kw in boy_keywords):
                return True
        return False

    if user_gender != "any":
        before = len(schemes)
        schemes = [s for s in schemes if not is_exclusive_for_other_gender(s, user_gender)]
        after = len(schemes)
        print(f"  Gender filter: {before} → {after} schemes after filtering for {user_gender.upper()}")

    return {**state, "retrieved_schemes": schemes}

# Node 4: Check if clarification needed 
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

# Node 5: Generate Response 
def generate_response(state: AgentState) -> AgentState:
    print("\n[Node 5] Generating response...")

    # Clarification needed 
    if state.get("needs_clarification"):
        msg = state['user_message'].lower()
        hindi_roman_words = ["main", "mujhe", "mera", "meri", "hoon", "hun", "hai", "hain",
                            "kya", "kaise", "mujko", "aap", "tum", "yojna", "yojana",
                            "se", "mein", "ka", "ki", "ke", "nahi", "chahiye", "batao",
                            "karo", "sarkari", "scheme", "bataiye", "kisan", "gaon"]
        is_devanagari = any(ord(c) > 127 for c in state['user_message'])
        is_roman_hindi = sum(1 for w in hindi_roman_words if w in msg.split()) >= 2
        lang = "Hindi" if (is_devanagari or is_roman_hindi) else "English"
        if lang == "Hindi":
            response = ("Namaste! Main aapko sarkari yojnaein dhundne mein madad kar sakta hoon.\n"
                        "Kripya apne baare mein bataiye:\n"
                        "- Aap kis state se hain?\n"
                        "- Aapka peshaa kya hai? (kisaan, student, mazdoor etc.)\n"
                        "- Kisi khaas yojana ki zaroorat hai? (shiksha, swasthya, krishi etc.)")
        else:
            response = ("Namaste! I can help you find government schemes you may be eligible for.\n"
                        "Could you tell me a bit about yourself?\n"
                        "- Which state are you from?\n"
                        "- What is your occupation? (farmer, student, worker etc.)\n"
                        "- Any specific type of scheme you are looking for? (education, health, agriculture etc.)")
        return {**state, "final_response": response}
    
    

    schemes = state.get("retrieved_schemes") or []
    entities = state.get("entities") or {}
    intent = state.get("intent", "find_scheme")

    user_caste      = entities.get("caste", "any")
    user_state      = entities.get("state", "any")
    user_occupation = entities.get("occupation", "any")
    user_income     = entities.get("income")
    user_age        = entities.get("age")
    user_gender     = entities.get("gender", "any")

    # Detect language from current message
    msg = state['user_message'].lower()
    hindi_roman_words = ["main", "mujhe", "mera", "meri", "hoon", "hun", "hai", "hain",
                        "kya", "kaise", "mujko", "aap", "tum", "yojna", "yojana",
                        "se", "mein", "ka", "ki", "ke", "nahi", "chahiye", "batao",
                        "karo", "sarkari", "scheme", "bataiye", "kisan", "gaon"]
    is_devanagari = any(ord(c) > 127 for c in state['user_message'])
    is_roman_hindi = sum(1 for w in hindi_roman_words if w in msg.split()) >= 2
    lang = "Hindi" if (is_devanagari or is_roman_hindi) else "English"

    # Translate last response 
    if intent == "translate":
        history = state.get("conversation_history") or []
        last_bot_response = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_bot_response = msg.get("content", "")
                break

        if last_bot_response:
            msg_lower = state['user_message'].lower()
            hindi_triggers = ["hindi", "हिंदी", "anuvad", "hindi mein", "hindi me"]
            english_triggers = ["english", "angrezi", "english mein"]

            if any(w in msg_lower for w in hindi_triggers):
                target_lang = "Hindi"
            elif any(w in msg_lower for w in english_triggers):
                target_lang = "English"
            else:
                target_lang = "Hindi"  # default if just "translate" said

            prompt = f"""Translate the following text to {target_lang}.
    Rules:
    - Keep all scheme names exactly as-is (e.g. "Mukhyamantri Jankalyan Yojana")
    - Keep all amounts exactly as-is (e.g. ₹50,000)
    - Keep all step numbers and document names as-is
    - Do NOT add any new information
    - Do NOT remove any information
    - ONLY translate the language, nothing else

    Text to translate:
    {last_bot_response}"""

            response = llm.invoke([HumanMessage(content=prompt)])
            return {**state, "final_response": response.content.strip()}

    # get_details: User asking about a specific scheme 
    if intent == "get_details":
        msg_lower = state["user_message"].lower()
        mentioned_scheme = None

        # First: match by words in current message vs scheme names
        for s in schemes:
            scheme_name_lower = s.get("scheme_name", "").lower()
            if any(word in scheme_name_lower for word in msg_lower.split() if len(word) > 4):
                mentioned_scheme = s
                break

        # Second: scan conversation history for last mentioned scheme
        if not mentioned_scheme:
            history = state.get("conversation_history") or []
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    for s in schemes:
                        scheme_name_lower = s.get("scheme_name", "").lower()
                        if any(word in content for word in scheme_name_lower.split() if len(word) > 5):
                            mentioned_scheme = s
                            break
                if mentioned_scheme:
                    break

        if mentioned_scheme:
            prompt = f"""You are YojnaSahay, a helpful assistant for Indian government schemes.
The user is asking for more details about a specific scheme. Give complete, thorough information.

Scheme Details:
Name: {mentioned_scheme.get('scheme_name')}
Benefits: {mentioned_scheme.get('benefits', '')[:1000]}
Eligibility (FULL - show every condition): {mentioned_scheme.get('eligibility_raw', '')[:1000]}
How to Apply (FULL - show every step): {mentioned_scheme.get('how_to_apply', '')[:1000]}
Documents Required (FULL - show every document): {mentioned_scheme.get('documents_required', '')[:600]}
Category: {mentioned_scheme.get('category')}
Level: {mentioned_scheme.get('level')}

User Profile:
- State: {user_state}
- Caste/Category: {user_caste}
- Occupation: {user_occupation}
- Age: {user_age}
- Annual Family Income: {user_income}

User message: "{state['user_message']}"

Instructions:
- Focus ONLY on this one scheme, do not mention other schemes.
- ALWAYS show ALL sections completely: Benefits, Eligibility, How to Apply, Documents Required.
- BENEFITS: Copy the EXACT benefits text word for word as given above. Do NOT paraphrase, summarize, or rewrite it. Show every amount, every category, every condition exactly as written.
- ELIGIBILITY: Show ALL eligibility conditions completely, not just the first one. Never cut short.
- HOW TO APPLY: Show ALL steps completely.
- DOCUMENTS: Show ALL documents listed completely.
- Do NOT filter benefits by user's domain — show the COMPLETE benefits so user can see all amounts.
- LANGUAGE RULE: Respond in {lang} only. Do NOT mix languages.
- Be conversational and friendly, add a brief intro line before showing details.
- At the end, ask if they want to know anything else about this scheme."""

            response = llm.invoke([HumanMessage(content=prompt)])
            final_response = response.content.strip()
            print(f"  Response generated ({len(final_response)} chars)")
            return {**state, "final_response": final_response}

    # find_scheme / fallback: List relevant schemes 
    schemes_text = ""
    for i, s in enumerate(schemes[:6], 1):
        schemes_text += f"""
Scheme {i}: {s.get('scheme_name')}
Benefits: {s.get('benefits', '')[:600]}
Eligibility: {s.get('eligibility_raw', '')[:250]}
How to Apply: {s.get('how_to_apply', '')[:150]}
---"""

    history_text = ""
    for msg in (state.get("conversation_history") or [])[-4:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    prompt = f"""You are YojnaSahay, a helpful assistant that helps Indian citizens find government schemes.
Be conversational, simple, and clear.

User profile:
- State: {user_state}
- Caste/Category: {user_caste}
- Gender: {user_gender}
- Occupation: {user_occupation}
- Age: {user_age}
- Annual Family Income: {user_income}

Relevant government schemes found:
{schemes_text}

Conversation so far:
{history_text}

Current user message: "{state['user_message']}"

STRICT Instructions:
- LANGUAGE RULE: Respond in {lang} only. Do NOT mix languages.
- CASTE RULE: User is {user_caste} category. STRICTLY skip any scheme meant only for SC/ST/OBC if user is General category, and vice versa.
- GENDER RULE: User is {user_gender}. STRICTLY skip any scheme meant only for girls/women if user is male, and vice versa. Never recommend girl-specific schemes to male users.
- INCOME RULE: If user's income is known and exceeds a scheme's income limit, skip that scheme.
- Only recommend schemes the user ACTUALLY qualifies for based on their full profile.
- Recommend 2-3 most relevant matching schemes only.
- For each scheme mention: exact benefit amounts SPECIFIC to the user's course/occupation.
- IMPORTANT: Benefits tables list amounts like "Rs. X/- CourseName". The amount BEFORE "Professional Graduation Course (BE, BTech)" is for B.Tech students. The amount before "Diploma Courses" is for diploma students. Never mix these up.
- Always match the user's course type to the correct row in the benefits table.
- Key eligibility conditions and how to apply briefly.
- If none of the retrieved schemes match the user's profile, say so honestly and ask for more details.
- Ask if they want more details about any specific scheme."""

    response = llm.invoke([HumanMessage(content=prompt)])
    final_response = response.content.strip()

    print(f"  Response generated ({len(final_response)} chars)")
    return {**state, "final_response": final_response}

# Build the Graph
def build_pipeline():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent",       detect_intent)
    graph.add_node("extract_entities",    extract_entities)
    graph.add_node("retrieve_schemes",    retrieve_schemes)
    graph.add_node("check_clarification", check_clarification)
    graph.add_node("generate_response",   generate_response)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent",       "extract_entities")
    graph.add_edge("extract_entities",    "retrieve_schemes")
    graph.add_edge("retrieve_schemes",    "check_clarification")
    graph.add_edge("check_clarification", "generate_response")
    graph.add_edge("generate_response",   END)

    return graph.compile()

pipeline = build_pipeline()

# Main Chat Function
def chat(user_message: str, conversation_history: List[dict] = []) -> str:
    state = AgentState(
        user_message=user_message,
        conversation_history=conversation_history,
        intent=None,
        entities=None,
        retrieved_schemes=None,
        final_response=None,
        needs_clarification=None
    )
    result = pipeline.invoke(state)
    return result["final_response"]

# Interactive Chat Loop
if __name__ == "__main__":
    print("=" * 60)
    print("YojnaSahay - Government Scheme Assistant")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = []
            print("Conversation reset!")
            continue

        response = chat(user_input, history)
        print(f"\nYojnaSahay: {response}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})