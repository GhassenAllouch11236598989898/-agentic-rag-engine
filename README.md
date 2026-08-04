<p align="center">
  <h1 align="center">🚀 Agentic RAG Engine</h1>
  <p align="center">
    <strong>Enterprise-Grade Retrieval-Augmented Generation with Local LLM & Vector Store</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Pydantic_AI-0.7+-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI" />
    <img src="https://img.shields.io/badge/pgvector-PostgreSQL_17-336791?logo=postgresql&logoColor=white" alt="pgvector" />
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  </p>
</p>

---

A production-ready **Agentic RAG** system that combines an autonomous AI agent with hybrid vector search and document ingestion. Designed to run **fully locally** with Ollama and SentenceTransformers — no paid API keys required — with seamless fallback to OpenAI when needed.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client / curl / UI                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │  HTTP / SSE
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API (:8000)                      │
│  ┌──────────┐  ┌───────────────┐  ┌───────────────────────────┐  │
│  │  /chat    │  │ /chat/stream  │  │ /search/vector | /hybrid  │  │
│  └────┬─────┘  └───────┬───────┘  └────────────┬──────────────┘  │
│       └────────────┬────┘                       │                │
│                    ▼                            │                │
│  ┌─────────────────────────────┐                │                │
│  │   Pydantic AI Agent         │ ◄──────────────┘                │
│  │   (tool-augmented reasoning)│                                 │
│  └──────────┬──────────────────┘                                 │
│             │  Tools: vector_search, hybrid_search,              │
│             │         get_document, list_documents               │
└─────────────┼────────────────────────────────────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐    ┌──────────────┐
│  LLM     │    │  Embeddings  │
│ Provider │    │  Provider    │
│          │    │              │
│ • Ollama │    │ • Sentence   │
│ • OpenAI │    │   Transformers│
└──────────┘    │ • OpenAI     │
                └──────┬───────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PostgreSQL + pgvector │
          │  (vector(384) / 1536)  │
          │                        │
          │  • Cosine similarity   │
          │  • Full-text search    │
          │  • Hybrid ranking      │
          └────────────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| 🤖 **Agentic Workflows** | Pydantic AI agent autonomously selects the right search tool per query |
| 🏠 **Local-First LLM** | Ollama integration (Llama 3.1, Mistral, etc.) — no API keys needed |
| 🔐 **OpenAI Fallback** | Seamless switch to GPT-4o-mini if `OPENAI_API_KEY` is provided |
| 🔍 **Hybrid Search** | Combines vector cosine similarity with PostgreSQL full-text search |
| ⚡ **Local Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers — fast and free |
| 🌊 **Real-Time Streaming** | Server-Sent Events for live token-by-token responses |
| 📄 **PDF Ingestion** | Docling-powered extraction: text, tables, images, OCR |
| 💬 **Session Memory** | Conversation history with automatic context injection |
| 🐳 **Dockerized** | One-command deployment with Docker Compose |

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- (Optional) [Ollama](https://ollama.ai/) for local LLM inference

### 1. Clone & Configure

```bash
git clone https://github.com/your-username/agentic-rag-engine.git
cd agentic-rag-engine

cp .env.example .env
# Edit .env with your preferred settings
```

### 2. Launch with Docker Compose

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 17 + pgvector** on port `5432`
- **FastAPI API** on port `8000`

### 3. (Optional) Start Ollama

If you want local LLM inference, uncomment the `ollama` service in `docker-compose.yml`, or run Ollama separately:

```bash
ollama pull llama3.1:8b
ollama serve
```

### 4. Ingest Documents

Place your PDF files in the `documents/` folder, then:

```bash
docker compose exec api python -m app.ingestion.ingest --documents documents/
```

### 5. Start Chatting!

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings in the uploaded documents?"}'
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/chat` | Single chat message |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `POST` | `/search/vector` | Vector similarity search |
| `POST` | `/search/hybrid` | Hybrid search |
| `GET` | `/documents` | List ingested documents |
| `GET` | `/sessions/{id}` | Session history |

### Example: Chat

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the main topics from all documents",
    "search_type": "hybrid"
  }' | python -m json.tool
```

**Response:**

```json
{
  "message": "Based on the documents in the knowledge base, the main topics are ...",
  "session_id": "a1b2c3d4-...",
  "tools_used": [
    {
      "tool_name": "hybrid_search",
      "args": {"query": "main topics summary", "limit": 10}
    }
  ],
  "metadata": {"search_type": "hybrid"}
}
```

### Example: Streaming Chat

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Give a detailed explanation of the findings"}'
```

### Example: Vector Search

```bash
curl -s -X POST http://localhost:8000/search/vector \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning techniques",
    "limit": 5
  }' | python -m json.tool
```

## Configuration

All configuration is managed through environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LLM_MODEL` | `llama3.1:8b` | Model name for Ollama |
| `EMBEDDING_PROVIDER` | `local` | `local` (SentenceTransformers) or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_DIM` | `384` | Must match model output dimension |
| `OPENAI_API_KEY` | — | Required only when using OpenAI providers |
| `APP_PORT` | `8000` | FastAPI port |
| `DB_HOST` | `postgres` | PostgreSQL hostname |

> **⚠️ Important:** The `EMBEDDING_DIM` value must match the dimension in `sql/schema.sql`. If you switch to OpenAI embeddings (`text-embedding-3-small` → 1536), update both the `.env` and the schema.

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | [Pydantic AI](https://ai.pydantic.dev/) |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **LLM** | [Ollama](https://ollama.ai/) / OpenAI |
| **Embeddings** | [SentenceTransformers](https://sbert.net/) / OpenAI |
| **Vector DB** | PostgreSQL 17 + [pgvector](https://github.com/pgvector/pgvector) |
| **PDF Processing** | [Docling](https://github.com/DS4SD/docling) |
| **Text Splitting** | [LangChain](https://langchain.readthedocs.io/) |
| **Containerization** | Docker + Docker Compose |

## Project Structure

```
├── app/
│   ├── __init__.py          # Package version
│   ├── api.py               # FastAPI endpoints & lifespan
│   ├── agent.py             # Pydantic AI agent + tools
│   ├── models.py            # Pydantic v2 request/response models
│   ├── prompts.py           # System prompt templates
│   ├── providers.py         # Dual LLM/Embedding provider (Ollama/OpenAI)
│   ├── tools.py             # Search & retrieval tool implementations
│   ├── db_utils.py          # PostgreSQL connection pool & queries
│   └── ingestion/
│       ├── ingest.py        # Document ingestion pipeline
│       ├── chunker.py       # Semantic & recursive text splitting
│       └── extract_files.py # PDF extraction via Docling
├── sql/
│   └── schema.sql           # pgvector schema + search functions
├── documents/               # Drop your PDFs here
├── tests/                   # Test suite
├── docker-compose.yml       # Orchestration
├── Dockerfile               # Multi-stage Python 3.12 build
├── pyproject.toml            # Project metadata & dependencies
└── .env.example             # Configuration template
```

## Development

### Local Setup (without Docker)

```bash
# Install uv (fast Python package manager)
pip install uv

# Install dependencies
uv pip install -r pyproject.toml

# Start PostgreSQL with pgvector (e.g., via Docker)
docker run -d --name pgvector \
  -e POSTGRES_DB=vector_db \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Run the API
uvicorn app.api:app --reload --port 8000
```

### Running Tests

```bash
pytest
```

## License

MIT
