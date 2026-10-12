"""Command line entry point:  python -m paper_rag.ingest [--force]"""

import argparse
import logging

from paper_rag.config import get_settings
from paper_rag.ingest.manifest import load_manifest
from paper_rag.ingest.pipeline import index_directory
from paper_rag.llm.embeddings import OpenAIEmbedder, embedding_cost_nok
from paper_rag.store.qdrant_store import QdrantStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index PDFs into the vector store.")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--manifest", default="data/papers.csv")
    parser.add_argument("--force", action="store_true", help="re-index even unchanged papers")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    embedder = OpenAIEmbedder.from_settings(settings)
    store = QdrantStore.from_settings(settings)

    report = index_directory(
        args.pdf_dir,
        store,
        embedder,
        manifest=load_manifest(args.manifest),
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        force=args.force,
    )

    cost = embedding_cost_nok(report.embedding_tokens, settings)
    print(
        f"\nindexed {len(report.indexed)} | updated {len(report.updated)} | "
        f"skipped {len(report.skipped)} | failed {len(report.failed)}"
    )
    print(
        f"{report.chunks_indexed} chunks, {report.embedding_tokens} tokens "
        f"(~{cost:.2f} NOK), {report.seconds:.1f} s, {store.count()} chunks in the store"
    )
    for name, reason in report.failed.items():
        print(f"  FAILED {name}: {reason}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
