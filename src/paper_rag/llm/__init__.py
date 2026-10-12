"""LLM providers (embeddings and chat) behind small interfaces."""

from paper_rag.llm.chat import ChatResult, FakeChat, LLMClient, OpenAIChat, chat_cost_nok
from paper_rag.llm.embeddings import (
    Embedder,
    EmbeddingResult,
    HashingEmbedder,
    OpenAIEmbedder,
    embedding_cost_nok,
)

__all__ = [
    "ChatResult",
    "Embedder",
    "EmbeddingResult",
    "FakeChat",
    "HashingEmbedder",
    "LLMClient",
    "OpenAIChat",
    "OpenAIEmbedder",
    "chat_cost_nok",
    "embedding_cost_nok",
]
