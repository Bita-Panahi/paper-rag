"""Per-paper metadata that cannot be read reliably from the PDF itself (year, own work)."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class PaperInfo:
    year: int | None = None
    own_paper: bool = False
    title: str = ""  # used only for recognising paper names in questions
    aliases: tuple[str, ...] = ()  # short names people use, e.g. "CIC 2024"


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, PaperInfo]:
    """Read papers.csv (columns: file_name, title, year, own_paper). Missing file -> empty."""
    path = Path(path)
    if not path.exists():
        return {}
    papers: dict[str, PaperInfo] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [(n or "").strip().lower() for n in reader.fieldnames or []]
        for line_no, row in enumerate(reader, start=2):
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            raw_year = (row.get("year") or "").strip()
            try:
                year = int(raw_year) if raw_year else None
            except ValueError:
                raise ManifestError(
                    f"{path.name} line {line_no}: year '{raw_year}' for '{file_name}' "
                    "is not a number"
                ) from None
            if year is not None and not 1900 <= year <= 2100:
                raise ManifestError(f"{path.name} line {line_no}: year {year} is out of range")
            own = (row.get("own_paper") or "").strip().lower() in _TRUE
            aliases = tuple(
                a.strip() for a in re.split(r"[;|,]", row.get("aliases") or "") if a.strip()
            )
            papers[file_name] = PaperInfo(
                year=year,
                own_paper=own,
                title=(row.get("title") or "").strip(),
                aliases=aliases,
            )
    return papers
