import pymupdf
import pytest


@pytest.fixture
def make_pdf():
    """Build a small text PDF. Each sentence becomes its own text block, 40 pt apart."""

    def _make(path, sentences: list[str], per_page: int = 12) -> None:
        doc = pymupdf.open()
        for start in range(0, len(sentences), per_page):
            page = doc.new_page()
            for i, sentence in enumerate(sentences[start : start + per_page]):
                page.insert_text((72, 100 + 40 * i), sentence, fontsize=11)
        doc.save(path)
        doc.close()

    return _make
