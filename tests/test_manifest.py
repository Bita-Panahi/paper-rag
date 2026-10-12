import pytest

from paper_rag.ingest.manifest import ManifestError, load_manifest


def write(tmp_path, text):
    path = tmp_path / "papers.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_year_and_own_paper(tmp_path):
    path = write(
        tmp_path,
        'file_name,title,year,own_paper\na.pdf,Title A,2024,true\nb.pdf,"Title, with comma",,\n',
    )
    manifest = load_manifest(path)
    assert manifest["a.pdf"].year == 2024 and manifest["a.pdf"].own_paper
    assert manifest["b.pdf"].year is None and not manifest["b.pdf"].own_paper


def test_missing_file_gives_empty_manifest(tmp_path):
    assert load_manifest(tmp_path / "nope.csv") == {}


def test_excel_bom_is_accepted(tmp_path):
    path = tmp_path / "papers.csv"
    body = "file_name,title,year,own_paper\na.pdf,T,2020,\n"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode())  # UTF-8 byte order mark, as Excel writes it
    assert load_manifest(path)["a.pdf"].year == 2020


@pytest.mark.parametrize("bad_year", ["abc", "20x4", "1850", "3000"])
def test_bad_year_fails_loudly(tmp_path, bad_year):
    path = write(tmp_path, f"file_name,title,year,own_paper\na.pdf,T,{bad_year},\n")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_title_and_aliases_are_read(tmp_path):
    path = tmp_path / "papers.csv"
    path.write_text(
        "file_name,title,year,own_paper,aliases\n"
        'a.pdf,Some Title,2024,true,"CIC 2024; Panahi paper|handheld"\n'
        "b.pdf,,,,\n",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    assert manifest["a.pdf"].title == "Some Title"
    assert manifest["a.pdf"].aliases == ("CIC 2024", "Panahi paper", "handheld")
    assert manifest["b.pdf"].aliases == ()
