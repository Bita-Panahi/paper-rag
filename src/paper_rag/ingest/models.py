"""Plain data containers passed between the ingestion steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Cleaned text of one PDF page (1-based page number)."""

    number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed PDF: stable identity (content hash), title and cleaned pages."""

    path: str
    file_name: str
    sha256: str
    title: str
    page_count: int
    pages: list[PageText]


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage plus the metadata needed for filtering and citations."""

    doc_id: str  # sha256 of the source file
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    title: str
    file_name: str

    @property
    def chunk_id(self) -> str:
        """Deterministic id: the same file and settings always give the same ids."""
        return f"{self.doc_id[:16]}-{self.chunk_index:04d}"
