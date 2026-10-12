"""PDF -> ParsedDocument: hash, title, and cleaned text per page."""

import hashlib
import re
from pathlib import Path

import pymupdf

from paper_rag.ingest.cleaning import build_vocabulary, clean_block
from paper_rag.ingest.models import PageText, ParsedDocument
from paper_rag.ingest.titles import resolve_title

_REFERENCES_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE)


def file_sha256(path: str | Path) -> str:
    """Fingerprint of the file bytes. Same hash means nothing to re-index."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_blocks(page: pymupdf.Page, margin_ratio: float) -> list[str]:
    """Raw text blocks of a page, in the PDF's own reading order, minus header/footer bands."""
    height = page.rect.height
    top_limit = height * margin_ratio
    bottom_limit = height * (1 - margin_ratio)
    blocks: list[str] = []
    for _x0, y0, _x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 0 = text, 1 = image
            continue
        if y1 <= top_limit or y0 >= bottom_limit:  # lies entirely in the header/footer band
            continue
        blocks.append(text)
    return blocks


def parse_pdf(
    path: str | Path,
    *,
    margin_ratio: float = 0.05,
    drop_references: bool = True,
) -> ParsedDocument:
    path = Path(path)
    with pymupdf.open(path) as doc:
        metadata_title = (doc.metadata or {}).get("title")
        raw_pages = [_page_blocks(page, margin_ratio) for page in doc]
        page_count = len(doc)

    vocabulary = build_vocabulary("\n".join(b for blocks in raw_pages for b in blocks))

    pages: list[PageText] = []
    in_references = False
    for number, blocks in enumerate(raw_pages, start=1):
        paragraphs: list[str] = []
        for raw in blocks:
            paragraph = clean_block(raw, vocabulary)
            if sum(c.isalpha() for c in paragraph) < 3:  # page numbers, stray digits
                continue
            if drop_references and _REFERENCES_HEADING.match(paragraph):
                in_references = True
            if in_references:
                break
            paragraphs.append(paragraph)
        if paragraphs:
            pages.append(PageText(number=number, text="\n\n".join(paragraphs)))
        if in_references:
            break

    return ParsedDocument(
        path=str(path),
        file_name=path.name,
        sha256=file_sha256(path),
        title=resolve_title(metadata_title, path.name),
        page_count=page_count,
        pages=pages,
    )
