"""The repeatable indexing pipeline: PDFs -> chunks -> embeddings -> vector store.

Running it twice does no extra work. A paper is re-processed only when its content, its
metadata (year, own_paper) or the indexing settings (chunk size, overlap, embedding model)
changed since it was stored.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from paper_rag.ingest.chunker import chunk_document
from paper_rag.ingest.manifest import PaperInfo
from paper_rag.ingest.parser import file_sha256, parse_pdf
from paper_rag.llm.embeddings import Embedder
from paper_rag.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)  # new papers
    updated: list[str] = field(default_factory=list)  # changed papers, re-indexed
    skipped: list[str] = field(default_factory=list)  # unchanged, nothing done
    failed: dict[str, str] = field(default_factory=dict)  # file name -> reason
    chunks_indexed: int = 0
    embedding_tokens: int = 0
    seconds: float = 0.0


def make_fingerprint(
    sha256: str,
    *,
    chunk_size: int,
    overlap: int,
    model: str,
    year: int | None = None,
    own_paper: bool = False,
) -> str:
    """Identifies exactly what was stored: file content, metadata and the settings used."""
    return f"{sha256}|{chunk_size}|{overlap}|{model}|{year}|{own_paper}"


def index_directory(
    pdf_dir: str | Path,
    store: QdrantStore,
    embedder: Embedder,
    *,
    manifest: dict[str, PaperInfo] | None = None,
    chunk_size: int = 800,
    overlap: int = 100,
    force: bool = False,
) -> IngestReport:
    manifest = manifest or {}
    report = IngestReport()
    started = time.perf_counter()
    stored = store.indexed_documents()

    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        name = pdf.name
        try:
            info = manifest.get(name, PaperInfo())
            fingerprint = make_fingerprint(
                file_sha256(pdf),
                chunk_size=chunk_size,
                overlap=overlap,
                model=embedder.model,
                year=info.year,
                own_paper=info.own_paper,
            )
            if not force and stored.get(name) == fingerprint:
                report.skipped.append(name)
                logger.info("skip %s (unchanged)", name)
                continue

            parsed = parse_pdf(pdf)
            chunks = chunk_document(parsed, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                report.failed[name] = "no extractable text (scanned PDF?)"
                logger.warning("fail %s: no extractable text", name)
                continue

            result = embedder.embed([c.text for c in chunks])

            if name in stored:
                store.delete_by_file_name(name)  # drop the old version before inserting the new
            store.upsert_document(
                chunks,
                result.vectors,
                year=info.year,
                own_paper=info.own_paper,
                fingerprint=fingerprint,
            )
        except Exception as exc:  # one broken file must not stop the whole run
            report.failed[name] = f"{type(exc).__name__}: {exc}"
            logger.exception("fail %s", name)
            continue

        (report.updated if name in stored else report.indexed).append(name)
        report.chunks_indexed += len(chunks)
        report.embedding_tokens += result.total_tokens
        logger.info("index %s: %d chunks, %d tokens", name, len(chunks), result.total_tokens)

    report.seconds = time.perf_counter() - started
    return report
