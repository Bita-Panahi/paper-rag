"""Run the API locally: python -m paper_rag.api  (then open http://localhost:8000/docs)."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("paper_rag.api.app:app", host="127.0.0.1", port=8000, reload=False)
