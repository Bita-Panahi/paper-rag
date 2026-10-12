"""Vector storage and metadata filtering on top of Qdrant."""

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from paper_rag.config import Settings
from paper_rag.ingest.models import Chunk
from paper_rag.store.models import SearchFilters, SearchHit

_UPSERT_BATCH = 128
_PAYLOAD_INDEXES = {
    "year": models.PayloadSchemaType.INTEGER,
    "own_paper": models.PayloadSchemaType.BOOL,
    "file_name": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or integers. uuid5 gives the same UUID for the same chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def to_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[models.Condition] = []
    if filters.year_min is not None or filters.year_max is not None:
        must.append(
            models.FieldCondition(
                key="year", range=models.Range(gte=filters.year_min, lte=filters.year_max)
            )
        )
    if filters.own_paper is not None:
        must.append(
            models.FieldCondition(key="own_paper", match=models.MatchValue(value=filters.own_paper))
        )
    if filters.file_name is not None:
        must.append(
            models.FieldCondition(key="file_name", match=models.MatchValue(value=filters.file_name))
        )
    if filters.file_names:
        must.append(
            models.FieldCondition(
                key="file_name", match=models.MatchAny(any=list(filters.file_names))
            )
        )
    return models.Filter(must=must) if must else None


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        if settings.qdrant_url == ":memory:":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60
            )
        return cls(client, settings.qdrant_collection)

    @classmethod
    def in_memory(cls, collection: str = "test") -> "QdrantStore":
        return cls(QdrantClient(":memory:"), collection)

    # ---- collection -------------------------------------------------------------------
    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors.size
            if existing != dim:
                raise ValueError(
                    f"Collection '{self.collection}' has vectors of size {existing}, but the "
                    f"embedder produces {dim}. Use another collection name or delete it."
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field, schema in _PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    # ---- write ------------------------------------------------------------------------
    def upsert_document(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        year: int | None,
        own_paper: bool,
        fingerprint: str,
    ) -> None:
        """Insert or overwrite all chunks of one paper. Same chunk id means same point."""
        self.ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=point_id(f"{chunk.file_name}/{chunk.chunk_id}"),
                vector=list(vector),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "title": chunk.title,
                    "file_name": chunk.file_name,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "year": year,
                    "own_paper": own_paper,
                    "index_fingerprint": fingerprint,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                self.collection, points=points[start : start + _UPSERT_BATCH], wait=True
            )

    def delete_by_file_name(self, file_name: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name", match=models.MatchValue(value=file_name)
                        )
                    ]
                )
            ),
            wait=True,
        )

    # ---- read -------------------------------------------------------------------------
    def indexed_documents(self) -> dict[str, str]:
        """file name -> index fingerprint of what is currently stored."""
        found: dict[str, str] = {}
        if not self.client.collection_exists(self.collection):
            return found
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["file_name", "index_fingerprint"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                found[payload["file_name"]] = payload.get("index_fingerprint", "")
            if offset is None:
                return found

    def search(
        self, vector: Sequence[float], top_k: int = 5, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection):
            return []
        response = self.client.query_points(
            self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=to_qdrant_filter(filters),
            with_payload=True,
        )
        hits = []
        for point in response.points:
            p = point.payload or {}
            hits.append(
                SearchHit(
                    score=point.score,
                    chunk_id=p["chunk_id"],
                    text=p["text"],
                    title=p["title"],
                    file_name=p["file_name"],
                    page_start=p["page_start"],
                    page_end=p["page_end"],
                    year=p.get("year"),
                    own_paper=bool(p.get("own_paper", False)),
                )
            )
        return hits
