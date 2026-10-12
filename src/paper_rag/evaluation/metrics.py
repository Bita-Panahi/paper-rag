"""Small, readable metrics. No LLM judge: facts are checked as text, so results are repeatable."""

import math
import re
from collections.abc import Sequence


def normalize(text: str) -> str:
    """Lowercase, keep only letters and digits (line breaks, hyphens, symbols don't matter)."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def fact_recall(text: str, facts: Sequence[str]) -> float:
    """Share of the required facts that appear in the text (1.0 = all of them)."""
    if not facts:
        return 1.0
    haystack = normalize(text)
    return sum(normalize(fact) in haystack for fact in facts) / len(facts)


def missing_facts(text: str, facts: Sequence[str]) -> list[str]:
    """The required facts that do NOT appear in the text (useful for debugging a failure)."""
    haystack = normalize(text)
    return [fact for fact in facts if normalize(fact) not in haystack]


def first_rank(file_names: Sequence[str], expected: Sequence[str]) -> int | None:
    """1-based position of the first result that comes from an expected paper, else None."""
    for rank, name in enumerate(file_names, start=1):
        if name in expected:
            return rank
    return None


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q between 0 and 100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]
