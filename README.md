# YojnaSahay 🇮🇳

**AI-powered Government Scheme Discovery Assistant for Indian Citizens**

YojnaSahay helps Indian citizens discover government schemes they are eligible for, through a conversational chatbot that supports both **Hindi and English**, including **voice input**.

🔗 **Live Demo:** [yojnasahay-4c64.vercel.app](https://yojnasahay-4c64.vercel.app)
🔗 **API:** [yojnasahay.onrender.com](https://yojnasahay.onrender.com)

---

##  Features

-  **Semantic Search** across 3,400+ government schemes using vector embeddings
-  **Voice Input** in Hindi and English via Groq Whisper API
-  **Bilingual** — responds in Hindi or English based on user's input language
-  **Smart Filtering** — caste, gender, income, occupation, and state-based eligibility filtering
-  **Multi-turn Conversations** — remembers context across the conversation (state, caste, income, etc.)
-  **Translate on demand** — "translate to hindi" re-translates the last response
-  **Detailed Scheme Info** — benefits, eligibility, how to apply, required documents

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend    │ ───▶ │   FastAPI     │ ───▶ │  LangGraph   │
│  (Vercel)    │      │   Backend     │      │   Pipeline   │
│  index.html  │ ◀─── │   (Render)    │ ◀─── │              │
└─────────────┘      └──────────────┘      └──────┬───────┘
                             │                      │
                             ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │  Groq Whisper │      │   Qdrant     │
                     │  (Voice STT)  │      │  Vector DB   │
                     └──────────────┘      │  (3400 schemes)│
                                            └─────────────┘
                                                    │
                                                    ▼
                                            ┌─────────────┐
                                            │  Groq LLM    │
                                            │ (llama-3.1)  │
                                            └─────────────┘
```

---

## 🧠 LangGraph Pipeline

The core of YojnaSahay is a **5-node stateful pipeline**:

| Node | Function | Description |
|------|----------|-------------|
| 1 | `detect_intent` | Classifies user intent — find scheme, get details, translate, or general |
| 2 | `extract_entities` | Extracts state, age, income, caste, gender, occupation from conversation |
| 3 | `retrieve_schemes` | Semantic search in Qdrant + post-filtering by caste/gender |
| 4 | `check_clarification` | Decides if the bot needs to ask follow-up questions |
| 5 | `generate_response` | Generates the final conversational response via Groq LLM |

```
detect_intent → extract_entities → retrieve_schemes → check_clarification → generate_response → END
```

State is shared across all nodes and persists across conversation turns via `conversation_history`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, Uvicorn |
| AI Orchestration | LangGraph, LangChain |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Vector Database | Qdrant Cloud |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Speech-to-Text | Groq Whisper API (`whisper-large-v3`) |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Containerization | Docker, Docker Compose |
| Deployment | Render (backend), Vercel (frontend), Qdrant Cloud (vector DB) |

---

##  Project Structure

```
yojnasahay/
├── main.py                 # FastAPI app — /chat, /transcribe, /reset endpoints
├── pipeline.py              # LangGraph pipeline — 5-node RAG workflow
├── ingest_schemes.py        # Script to ingest schemes into Qdrant
├── requirements.txt         # Python dependencies
├── runtime.txt               # Python version pin (3.11.9)
├── Dockerfile                # Container definition for backend
├── docker-compose.yml        # Local container orchestration
├── .dockerignore
├── .gitignore
├── render.yaml               # Render deployment config
└── frontend/
    ├── index.html            # Chat UI (Vercel static deployment)
    └── vercel.json
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- Groq API key
- Qdrant Cloud cluster (URL + API key)

### Setup

```bash
git clone https://github.com/aeshaladani/yojnasahay.git
cd yojnasahay

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Run Backend
```bash
uvicorn main:app --reload --port 8000
```

### Run Frontend
```bash
cd frontend
python -m http.server 3000
```
Open `http://localhost:3000` (set `API_BASE` in `index.html` to `http://localhost:8000` for local testing).

---

## 🐳 Running with Docker

```bash
docker-compose up -d
```

This builds the image and starts the backend container with environment variables loaded from `.env`, exposed on `http://localhost:8000`.

To stop:
```bash
docker-compose down
```

---

## 📡 API Endpoints

### `GET /`
Health check.
```json
{"status": "YojnaSahay API is running", "whisper": "groq-api"}
```

### `POST /chat`
```json
{
  "message": "I am a farmer in Bihar",
  "conversation_history": []
}
```
Returns the assistant's response along with updated conversation history.

### `POST /transcribe`
Accepts a multipart audio file (`webm`/`wav`/`mp3`), returns transcribed text and detected language using Groq Whisper API.

### `POST /reset`
Resets conversation context (client should send `conversation_history: []`).

---

##  Filtering Logic

- **Caste filter**: General category users won't see SC/ST/OBC-exclusive schemes, and vice versa
- **Gender filter**: Male users won't see women/girl-exclusive schemes (Kanyashree, Ladli, Beti Bachao, etc.)
- **Income filter**: Schemes with income caps below the user's stated income are excluded
- **State filter**: Schemes are matched against the user's state where applicable

---

##  Future Improvements

- Add conditional edges in LangGraph (skip retrieval for `get_details` intent)
- Persist conversation history server-side (currently client-managed)
- Add more regional languages beyond Hindi/English
- Cache embedding model in a separate microservice to reduce cold-start latency

---
