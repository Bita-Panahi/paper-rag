"""Recognise paper names in a question, so the search can be limited to those papers.

"What tools are used in the CIC 2024 paper?" mentions a paper. Searching all 15 papers equally
is wrong for such a question; limiting the search to that paper (a metadata filter) is right.
"""

import re
from collections.abc import Mapping
from pathlib import Path

from paper_rag.ingest.manifest import PaperInfo

_MIN_ALIAS_CHARS = 3


def _norm(text: str) -> str:
    """Lowercase; every run of non-alphanumerics becomes one space."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class PaperResolver:
    """Maps names found in a question (alias, title, file name) to paper file names."""

    def __init__(self, papers: Mapping[str, PaperInfo]) -> None:
        self._names: list[tuple[str, str]] = []  # (normalised name, file name)
        for file_name, info in papers.items():
            names = {file_name, Path(file_name).stem, info.title, *info.aliases}
            for name in names:
                normalised = _norm(name)
                if len(normalised) >= _MIN_ALIAS_CHARS:
                    self._names.append((normalised, file_name))

    def resolve(self, question: str) -> tuple[str, ...]:
        """File names of every paper the question names (empty if it names none)."""
        padded = f" {_norm(question)} "
        found: dict[str, None] = {}
        for name, file_name in self._names:
            if f" {name} " in padded:
                found[file_name] = None
        return tuple(found)
