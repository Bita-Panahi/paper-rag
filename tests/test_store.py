import pytest

from paper_rag.ingest.models import Chunk
from paper_rag.llm.embeddings import HashingEmbedder
from paper_rag.store import QdrantStore, SearchFilters

EMBED = HashingEmbedder(dim=64)


def make_chunks(file_name: str, texts: list[str], doc_id: str = "d" * 64) -> list[Chunk]:
    return [
        Chunk(
            doc_id,
            i,
            t,
            page_start=i + 1,
            page_end=i + 1,
            title=f"T {file_name}",
            file_name=file_name,
        )
        for i, t in enumerate(texts)
    ]


def add(store, file_name, texts, *, year, own, fingerprint="fp"):
    chunks = make_chunks(file_name, texts)
    store.upsert_document(
        chunks, EMBED.embed(texts).vectors, year=year, own_paper=own, fingerprint=fingerprint
    )


@pytest.fixture
def store():
    s = QdrantStore.in_memory()
    add(
        s,
        "old.pdf",
        ["goniochromatic coatings measured at many angles", "wavelet compression"],
        year=2015,
        own=False,
    )
    add(
        s,
        "new.pdf",
        ["neural network for goniochromatic appearance", "importance sampling"],
        year=2024,
        own=True,
    )
    return s


def q(text):
    return EMBED.embed([text]).vectors[0]


def test_search_returns_best_match_with_metadata(store):
    hit = store.search(q("goniochromatic coatings angles"), top_k=1)[0]
    assert hit.file_name == "old.pdf"
    assert hit.page_start == 1 and hit.year == 2015 and hit.own_paper is False
    assert hit.title == "T old.pdf"


def test_year_filter(store):
    hits = store.search(q("goniochromatic"), top_k=10, filters=SearchFilters(year_min=2020))
    assert hits and {h.file_name for h in hits} == {"new.pdf"}
    hits = store.search(q("goniochromatic"), top_k=10, filters=SearchFilters(year_max=2020))
    assert {h.file_name for h in hits} == {"old.pdf"}


def test_own_paper_and_file_filters(store):
    hits = store.search(q("goniochromatic"), top_k=10, filters=SearchFilters(own_paper=True))
    assert {h.file_name for h in hits} == {"new.pdf"}
    hits = store.search(q("goniochromatic"), top_k=10, filters=SearchFilters(file_name="old.pdf"))
    assert {h.file_name for h in hits} == {"old.pdf"}


def test_paper_without_year_never_matches_a_year_filter():
    s = QdrantStore.in_memory()
    add(s, "unknown.pdf", ["goniochromatic"], year=None, own=False)
    assert s.search(q("goniochromatic"), filters=SearchFilters(year_min=1900)) == []
    assert len(s.search(q("goniochromatic"))) == 1


def test_upsert_is_idempotent(store):
    before = store.count()
    add(
        store,
        "new.pdf",
        ["neural network for goniochromatic appearance", "importance sampling"],
        year=2024,
        own=True,
    )
    assert store.count() == before


def test_same_content_under_two_names_does_not_collide():
    s = QdrantStore.in_memory()
    add(s, "a.pdf", ["same text"], year=2020, own=False)
    add(s, "b.pdf", ["same text"], year=2020, own=False)
    assert s.count() == 2


def test_delete_by_file_name(store):
    store.delete_by_file_name("old.pdf")
    assert set(store.indexed_documents()) == {"new.pdf"}
    assert store.count() == 2


def test_indexed_documents_reports_fingerprints():
    s = QdrantStore.in_memory()
    add(s, "a.pdf", ["x y z"], year=2020, own=False, fingerprint="abc")
    assert s.indexed_documents() == {"a.pdf": "abc"}


def test_empty_store_is_handled():
    s = QdrantStore.in_memory()
    assert s.count() == 0
    assert s.indexed_documents() == {}
    assert s.search(q("anything")) == []
    s.delete_by_file_name("nothing.pdf")  # must not raise


def test_vector_size_mismatch_is_reported(store):
    with pytest.raises(ValueError, match="size 64"):
        store.ensure_collection(1536)


def test_filter_on_several_file_names():
    s = QdrantStore.in_memory()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        add(s, name, ["goniochromatic coatings measured at many angles"], year=2020, own=False)
    hits = s.search(
        EMBED.embed(["coatings"]).vectors[0], 10, SearchFilters(file_names=("a.pdf", "c.pdf"))
    )
    assert {h.file_name for h in hits} == {"a.pdf", "c.pdf"}
