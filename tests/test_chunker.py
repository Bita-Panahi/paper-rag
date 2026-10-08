from itertools import pairwise

import pytest

from paper_rag.ingest.chunker import chunk_document
from paper_rag.ingest.models import PageText, ParsedDocument


def make_doc(pages: list[str]) -> ParsedDocument:
    return ParsedDocument(
        path="p.pdf",
        file_name="p.pdf",
        sha256="a" * 64,
        title="T",
        page_count=len(pages),
        pages=[PageText(number=i, text=t) for i, t in enumerate(pages, start=1)],
    )


SENTENCES = [f"Sentence number {i} talks about goniochromatic pigments." for i in range(40)]


def test_chunks_respect_size_and_overlap():
    doc = make_doc([" ".join(SENTENCES)])
    chunks = chunk_document(doc, chunk_size=300, overlap=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    # consecutive chunks share their boundary sentence(s)
    for a, b in pairwise(chunks):
        last_sentence = a.text.split(". ")[-1]
        assert last_sentence in b.text


def test_no_text_is_lost():
    doc = make_doc([" ".join(SENTENCES)])
    chunks = chunk_document(doc, chunk_size=300, overlap=80)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in SENTENCES)


def test_pages_are_tracked():
    doc = make_doc(["First page sentence. " * 5, "Second page sentence. " * 5])
    chunks = chunk_document(doc, chunk_size=10_000, overlap=0)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_ids_are_deterministic_and_unique():
    doc = make_doc([" ".join(SENTENCES)])
    first = [c.chunk_id for c in chunk_document(doc, 300, 80)]
    second = [c.chunk_id for c in chunk_document(doc, 300, 80)]
    assert first == second
    assert len(set(first)) == len(first)


def test_overlong_sentence_is_split_by_words():
    doc = make_doc(["word " * 400])
    chunks = chunk_document(doc, chunk_size=200, overlap=0)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)


def test_empty_document_gives_no_chunks():
    assert chunk_document(make_doc([]), 300, 50) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(make_doc(["Hello world."]), size, overlap)
