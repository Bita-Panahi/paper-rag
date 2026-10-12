"""Shared test setup: a small in-memory RagService with a fake LLM."""

from paper_rag.config import Settings
from paper_rag.ingest.models import Chunk
from paper_rag.llm.chat import FakeChat
from paper_rag.llm.embeddings import HashingEmbedder
from paper_rag.retrieval.service import RagService
from paper_rag.store import QdrantStore

TEXTS = {
    "coating.pdf": [
        "goniochromatic coating measured at many angles",
        "pigment flakes shift colour",
    ],
    "neural.pdf": ["neural network learns a latent brdf representation"],
}


def build(reply="Coatings shift colour [1].", min_score=0.1, resolver=None, scope=True):
    settings = Settings(_env_file=None, min_score=min_score, top_k=3, scope_to_named_papers=scope)
    store, embedder, llm = QdrantStore.in_memory(), HashingEmbedder(dim=64), FakeChat(reply)
    for name, texts in TEXTS.items():
        chunks = [
            Chunk("d" * 64, i, t, i + 1, i + 1, f"Title {name}", name) for i, t in enumerate(texts)
        ]
        store.upsert_document(
            chunks,
            embedder.embed(texts).vectors,
            year=2020 if name == "coating.pdf" else 2024,
            own_paper=name == "coating.pdf",
            fingerprint="fp",
        )
    return RagService(store, embedder, llm, settings, resolver), llm
