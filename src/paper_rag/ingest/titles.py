"""Choose a usable title: PDF metadata is often empty or contains junk placeholders."""

import re
from pathlib import Path

_JUNK_MARKERS = ("microsoft word", "untitled", "article info", ".docx", ".doc", ".tex", "**")


def is_plausible_title(title: str) -> bool:
    t = title.strip()
    if not 8 <= len(t) <= 300:
        return False
    lowered = t.lower()
    return not any(marker in lowered for marker in _JUNK_MARKERS)


def prettify_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip()


def resolve_title(metadata_title: str | None, file_name: str) -> str:
    """Metadata title if it looks real, otherwise the cleaned-up file name."""
    if metadata_title and is_plausible_title(metadata_title):
        return metadata_title.strip()
    return prettify_filename(file_name)
