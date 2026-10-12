"""ParsedDocument -> overlapping, page-aware Chunks.

Text is split into sentences, then sentences are packed into chunks of at most
`chunk_size` characters. Each new chunk starts with the last few sentences of the
previous one (up to `overlap` characters), so no idea is cut off at a boundary.
"""

import re
from dataclasses import dataclass

from paper_rag.ingest.models import Chunk, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")

@dataclass(frozen=True)
class _Segment:
    page: int
    text: str


def _hard_split(text: str, size: int) -> list[str]:
    """Fallback for a 'sentence' longer than a whole chunk: split at word boundaries."""
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        extra = len(word) + (1 if current else 0)
        if current and length + extra > size:
            parts.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += extra
    if current:
        parts.append(" ".join(current))
    return parts


def _segments(doc: ParsedDocument, chunk_size: int) -> list[_Segment]:
    segments: list[_Segment] = []
    for page in doc.pages:
        for paragraph in page.text.split("\n\n"):
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= chunk_size:
                    segments.append(_Segment(page.number, sentence))
                else:
                    segments.extend(
                        _Segment(page.number, p) for p in _hard_split(sentence, chunk_size)
                    )
    return segments


def _joined(segments: list[_Segment]) -> str:
    return " ".join(s.text for s in segments)


def chunk_document(doc: ParsedDocument, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []

    def emit(current: list[_Segment]) -> None:
        chunks.append(
            Chunk(
                doc_id=doc.sha256,
                chunk_index=len(chunks),
                text=_joined(current),
                page_start=min(s.page for s in current),
                page_end=max(s.page for s in current),
                title=doc.title,
                file_name=doc.file_name,
            )
        )

    current: list[_Segment] = []
    for segment in _segments(doc, chunk_size):
        if current and len(_joined(current + [segment])) > chunk_size:
            emit(current)
            carry: list[_Segment] = []
            for previous in reversed(current):
                if len(_joined([previous] + carry)) > overlap:
                    break
                carry.insert(0, previous)
            while carry and len(_joined(carry + [segment])) > chunk_size:
                carry.pop(0)
            current = carry
        current.append(segment)
    if current:
        emit(current)
    return chunks
