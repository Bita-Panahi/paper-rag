"""Text cleaning for raw PDF extraction output.

PDFs store where to draw characters, not paragraphs, so extracted text has hard line
breaks inside sentences, words split with hyphens at line ends, and ligature glyphs.
"""

import re

_LINE_BREAK_HYPHEN = re.compile(r"([A-Za-z]+)-\n([a-z]+)")
_WORD = re.compile(r"[A-Za-z]+")
_SUSPENDED_HYPHEN_WORDS = {"and", "or", "to"}
_LIGATURES = str.maketrans(
    {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
)


def build_vocabulary(text: str) -> set[str]:
    """Words of the document that are NOT split across lines (lowercased).

    Used to decide what a line-end hyphen means: a hyphenation inside one word
    ("geo-" "metrical") or a real compound ("low-" "dimensional").
    """
    unsplit = _LINE_BREAK_HYPHEN.sub(" ", text)
    return {w.lower() for w in _WORD.findall(unsplit)}


def dehyphenate(text: str, vocabulary: set[str]) -> str:
    """Repair words split by a hyphen at the end of a line.

    Rules, in order:
      1. "interference-" "and ..." is a suspended hyphen: keep it, with a space.
      2. If the joined word occurs elsewhere in the paper, join it.
      3. If both halves are words on their own, it is a real compound: keep the hyphen.
      4. Otherwise (a half such as "geo" or "metrical" is not a word) it is a
         hyphenation: join it.
    """

    def repair(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        left_l, right_l = left.lower(), right.lower()
        if right_l in _SUSPENDED_HYPHEN_WORDS:
            return f"{left}- {right}"
        if left_l + right_l in vocabulary:
            return left + right
        if left_l in vocabulary and right_l in vocabulary:
            return f"{left}-{right}"
        return left + right

    text = _LINE_BREAK_HYPHEN.sub(repair, text)
    # "Cook-" / "Torrance": a hyphen before a capital or digit is always a real hyphen
    return re.sub(r"([A-Za-z])-\n(?=[A-Z0-9])", r"\1-", text)


def clean_block(text: str, vocabulary: set[str]) -> str:
    """Turn one extracted text block into a single clean paragraph."""
    text = text.replace("\u00ad", "").translate(_LIGATURES)
    text = dehyphenate(text, vocabulary)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
