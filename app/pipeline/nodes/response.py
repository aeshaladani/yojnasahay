from langchain_core.messages import HumanMessage
from app.pipeline.state import AgentState
from app.services.llm import llm

hindi_roman_words = ["main", "mujhe", "mera", "meri", "hoon", "hun", "hai", "hain",
                    "kya", "kaise", "mujko", "aap", "tum", "yojna", "yojana",
                    "se", "mein", "ka", "ki", "ke", "nahi", "chahiye", "batao",
                    "karo", "sarkari", "scheme", "bataiye", "kisan", "gaon"]

def detect_language(user_message: str) -> str:
    msg = user_message.lower()
    is_devanagari = any(ord(c) > 127 for c in user_message)
    is_roman_hindi = sum(1 for w in hindi_roman_words if w in msg.split()) >= 2
    return "Hindi" if (is_devanagari or is_roman_hindi) else "English"

# Node 5: Generate Response
def generate_response(state: AgentState) -> AgentState:
    print("\n[Node 5] Generating response...")

    # Clarification needed
    if state.get("needs_clarification"):
        lang = detect_language(state['user_message'])
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

    lang = detect_language(state['user_message'])

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
