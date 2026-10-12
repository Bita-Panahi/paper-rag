"""Types used to talk to the vector store."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchFilters:
    """Metadata conditions applied together (AND). None means 'do not filter on this'."""

    year_min: int | None = None
    year_max: int | None = None
    own_paper: bool | None = None
    file_name: str | None = None


@dataclass(frozen=True)
class SearchHit:
    score: float
    chunk_id: str
    text: str
    title: str
    file_name: str
    page_start: int
    page_end: int
    year: int | None
    own_paper: bool
