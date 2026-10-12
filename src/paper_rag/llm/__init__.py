"""LLM providers (embeddings now, chat completion later) behind small interfaces."""

from paper_rag.llm.embeddings import (
    Embedder,
    EmbeddingResult,
    HashingEmbedder,
    OpenAIEmbedder,
    embedding_cost_nok,
)

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "embedding_cost_nok",
]
