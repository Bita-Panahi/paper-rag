"""Embedding providers behind one small interface.

The rest of the code only knows `Embedder.embed(texts)`. Swapping OpenAI for Azure OpenAI
or another provider means adding a class here, not touching the pipeline.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from paper_rag.config import Settings


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class OpenAIEmbedder:
    """Embeddings from the OpenAI API (or any OpenAI-compatible client, e.g. AzureOpenAI)."""

    def __init__(self, client, model: str, batch_size: int = 64) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIEmbedder":
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is empty. Put it in your .env file.")
        client = OpenAI(api_key=settings.openai_api_key, max_retries=5)
        return cls(client, settings.embedding_model, settings.embedding_batch_size)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(list(item.embedding) for item in ordered)
            total_tokens += response.usage.total_tokens
        return EmbeddingResult(vectors=vectors, total_tokens=total_tokens)


class HashingEmbedder:
    """Tiny offline embedder (bag of hashed words). Free, deterministic, no network.

    Not semantically smart, but texts that share words get similar vectors, which is enough
    to test the whole pipeline and the vector store without calling an API.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model = f"hashing-{dim}"
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls += 1
        vectors = [self._vector(t) for t in texts]
        tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResult(vectors=vectors, total_tokens=tokens)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def embedding_cost_nok(total_tokens: int, settings: Settings) -> float:
    """Estimated cost in NOK from list prices (an estimate, not an invoice)."""
    usd = total_tokens / 1_000_000 * settings.embedding_price_per_million_usd
    return usd * settings.usd_to_nok
