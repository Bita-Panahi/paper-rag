# paper-rag

A small, tested retrieval-augmented generation (RAG) system for question answering over
scientific papers (PDF). Built as a backend API, not a notebook.

> Status: under construction. See the roadmap below.

## Architecture (planned)

```
PDFs -> ingest (parse -> chunk -> embed) -> Qdrant (vectors + metadata)
Question -> FastAPI -> retrieve (+ metadata filter) -> LLM -> answer + citations
                          \-> logs: latency, tokens, cost
```

## Roadmap

- [ ] Repeatable ingestion pipeline (skips unchanged PDFs)
- [ ] Vector storage with metadata filtering (Qdrant)
- [ ] Retrieval and answer-quality evaluation
- [ ] OpenAI integration behind a provider interface
- [ ] FastAPI backend (`/ingest`, `/ask`, `/health`)
- [ ] Docker
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment (Azure Container Apps)
- [ ] Logging, error handling, cost and latency measurement

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then add your OPENAI_API_KEY
pytest
```
