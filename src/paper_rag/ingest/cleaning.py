"""Text cleaning for raw PDF extraction output.

PDFs store where to draw characters, not paragraphs, so extracted text has hard line
breaks inside sentences, words split with hyphens at line ends, and ligature glyphs.
"""

import re

_LINE_BREAK_HYPHEN = re.compile(r"([A-Za-z]+)-\n([a-z]+)")
_WORD = re.compile(r"[A-Za-z]+")
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})


def build_vocabulary(text: str) -> set[str]:
    """Words of the document that are NOT split across lines (lowercased).

    Used to decide whether a line-end hyphen is a hyphenation ("func-" "tions") or a
    real compound ("low-" "dimensional"): if the joined word occurs elsewhere in the
    paper, it is a hyphenation.
    """
    unsplit = _LINE_BREAK_HYPHEN.sub(" ", text)
    return {w.lower() for w in _WORD.findall(unsplit)}


def dehyphenate(text: str, vocabulary: set[str]) -> str:
    def repair(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        joined = left + right
        return joined if joined.lower() in vocabulary else f"{left}-{right}"

    return _LINE_BREAK_HYPHEN.sub(repair, text)


def clean_block(text: str, vocabulary: set[str]) -> str:
    """Turn one extracted text block into a single clean paragraph."""
    text = text.replace("­", "").translate(_LIGATURES)
    text = dehyphenate(text, vocabulary)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
