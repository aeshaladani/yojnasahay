import json
import os
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# Config
COLLECTION_NAME = "schemes"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE       = 25
JSON_FILE        = "schemes_cleaned.json"

# Load Data 
print("Loading schemes...")
with open(JSON_FILE, "r", encoding="utf-8") as f:
    schemes = json.load(f)
print(f"Loaded {len(schemes)} schemes")

# Load Embedding Model 
print("\nLoading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded!")

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout =60
)

# Check if Collection Already Exists
existing = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME in existing:
    print(f"\nCollection '{COLLECTION_NAME}' already exists, skipping ingestion...")
    client.delete_collection(COLLECTION_NAME)
    import time
    time.sleep(2) 
    SKIP_INGESTION = False
else:
    SKIP_INGESTION = False
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created!")

# Helper: Build searchable text from scheme 
def build_text(scheme):
    # Repeat state multiple times so it gets weighted higher
    state = scheme.get("level", "")
    eligibility = scheme.get("eligibility", {})
    state_name = eligibility.get("state", "") if isinstance(eligibility, dict) else ""
    
    parts = [
        scheme.get("scheme_name", ""),
        scheme.get("scheme_name", ""),  # repeat name for weight
        scheme.get("description", "")[:500],
        scheme.get("benefits", "")[:300],
        scheme.get("eligibility_raw", "")[:300],
        " ".join(scheme.get("category", [])),
        " ".join(scheme.get("keywords", [])),
        scheme.get("tags", ""),
        state,
        state_name,
    ]
    return " ".join([p for p in parts if p]).strip()

# Helper: Build payload
def build_payload(scheme):
    return {
        "scheme_id":           scheme.get("scheme_id"),
        "scheme_name":         scheme.get("scheme_name"),
        "description":         scheme.get("description", "")[:1000],
        "benefits":            scheme.get("benefits", "")[:1000],
        "eligibility_raw":     scheme.get("eligibility_raw", "")[:1000],
        "eligibility":         scheme.get("eligibility", {}),
        "how_to_apply":        scheme.get("how_to_apply", "")[:1000],
        "documents_required":  scheme.get("documents_required", "")[:600],
        "category":            scheme.get("category", []),
        "level":               scheme.get("level", ""),
        "tags":                scheme.get("tags", ""),
        "keywords":            scheme.get("keywords", []),
        "slug":                scheme.get("slug", ""),
    }

# Ingest in Batches
if not SKIP_INGESTION:
    print(f"\nStarting ingestion of {len(schemes)} schemes in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(schemes), BATCH_SIZE), desc="Ingesting"):
        batch = schemes[i : i + BATCH_SIZE]
        texts = [build_text(s) for s in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=i + j,
                vector=embeddings[j].tolist(),
                payload=build_payload(batch[j])
            )
            for j in range(len(batch))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\n Ingestion complete! {len(schemes)} schemes stored in Qdrant")
else:
    print(" Using existing collection, no ingestion needed.")

# Quick Test Search
print("\n--- Running test search: 'financial help for MP students general category' ---")

query = "financial help for madhya pradesh students general category"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
).points

print(f"\nTop 3 results:")
for idx, r in enumerate(results):
    print(f"\n{idx+1}. {r.payload['scheme_name']}")
    print(f"   Score: {r.score:.3f}")
    print(f"   Category: {r.payload['category']}")
    print(f"   Benefits: {r.payload['benefits'][:100]}...")