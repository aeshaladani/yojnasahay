from qdrant_client.models import Filter, FieldCondition, MatchText
from app.pipeline.state import AgentState
from app.services.qdrant import qdrant, COLLECTION
from app.services.embedder import get_embedder
from app.filters.caste import apply_caste_filter
from app.filters.gender import apply_gender_filter

# Node 4: RAG Retrieval
def retrieve_schemes(state: AgentState) -> AgentState:
    print("\n[Node 4] Retrieving schemes from Qdrant...")

    entities = state.get("entities") or {}

    # Build search query
    query_parts = [state["user_message"]]
    if entities.get("occupation"): query_parts.append(entities["occupation"])
    if entities.get("category"):   query_parts.append(entities["category"])
    if entities.get("state"):      query_parts.append(entities["state"])
    if entities.get("caste"):      query_parts.append(entities["caste"])
    search_query = " ".join(query_parts)

    query_vector = get_embedder().encode(search_query).tolist()

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

    # Apply caste and gender filters
    user_caste = (entities.get("caste") or "any").lower()
    user_gender = (entities.get("gender") or "any").lower()

    schemes = apply_caste_filter(schemes, user_caste)
    schemes = apply_gender_filter(schemes, user_gender)

    return {**state, "retrieved_schemes": schemes}
