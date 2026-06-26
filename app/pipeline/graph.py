from langgraph.graph import StateGraph, END
from app.pipeline.state import AgentState
from app.pipeline.nodes.intent import detect_intent
from app.pipeline.nodes.entities import extract_entities
from app.pipeline.nodes.clarification import check_clarification
from app.pipeline.nodes.retrieval import retrieve_schemes
from app.pipeline.nodes.response import generate_response

def route_after_intent(state):
    if state.get("intent") in ["translate", "general"]:
        return "generate_response"
    return "extract_entities"

def build_pipeline():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent",       detect_intent)
    graph.add_node("extract_entities",    extract_entities)
    graph.add_node("check_clarification", check_clarification)
    graph.add_node("retrieve_schemes",    retrieve_schemes)
    graph.add_node("generate_response",   generate_response)

    graph.set_entry_point("detect_intent")

    # Conditional edge after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "generate_response": "generate_response",
            "extract_entities":  "extract_entities"
        }
    )

    # Normal flow for find_scheme / get_details / eligibility_check
    graph.add_edge("extract_entities",    "check_clarification")
    graph.add_edge("check_clarification", "retrieve_schemes")
    graph.add_edge("retrieve_schemes",    "generate_response")
    graph.add_edge("generate_response",   END)

    return graph.compile()

pipeline = build_pipeline()

def chat(user_message: str, conversation_history: list = []) -> str:
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