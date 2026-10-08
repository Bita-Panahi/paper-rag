import pymupdf

from paper_rag.ingest.parser import file_sha256, parse_pdf


def build_pdf(path, *, title: str = "") -> None:
    doc = pymupdf.open()
    page1 = doc.new_page()  # A4: 595 x 842 pt
    page1.insert_text((72, 20), "RUNNING HEADER 14", fontsize=9)
    page1.insert_text(
        (72, 120), "We study goniochromatic coatings under varying light.", fontsize=11
    )
    page1.insert_text((72, 200), "Results show strong angular dependence of colour.", fontsize=11)
    page1.insert_text((72, 830), "page footer 1", fontsize=9)
    page2 = doc.new_page()
    page2.insert_text(
        (72, 120), "A second page with further discussion of the method.", fontsize=11
    )
    page2.insert_text((72, 300), "References", fontsize=14)
    page2.insert_text(
        (72, 340), "[1] Someone. A cited paper that must not be indexed.", fontsize=11
    )
    doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


def test_parse_strips_header_footer_and_references(tmp_path):
    pdf = tmp_path / "My_Test_Paper.pdf"
    build_pdf(pdf)
    parsed = parse_pdf(pdf)

    assert parsed.page_count == 2
    assert parsed.file_name == "My_Test_Paper.pdf"
    assert parsed.title == "My Test Paper"  # empty metadata -> file name fallback
    text = "\n".join(p.text for p in parsed.pages)
    assert "goniochromatic coatings" in text
    assert "further discussion" in text
    assert "RUNNING HEADER" not in text
    assert "page footer" not in text
    assert "References" not in text
    assert "must not be indexed" not in text


def test_references_can_be_kept(tmp_path):
    pdf = tmp_path / "p.pdf"
    build_pdf(pdf)
    text = "\n".join(p.text for p in parse_pdf(pdf, drop_references=False).pages)
    assert "must not be indexed" in text


def test_metadata_title_is_used_when_plausible(tmp_path):
    pdf = tmp_path / "p.pdf"
    build_pdf(pdf, title="A Plausible Real Paper Title")
    assert parse_pdf(pdf).title == "A Plausible Real Paper Title"


def test_hash_is_stable_and_changes_with_content(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    build_pdf(a)
    build_pdf(b, title="Different Metadata Title")
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)
