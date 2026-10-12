import pytest

from paper_rag.ingest.manifest import ManifestError, PaperInfo, load_manifest


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
    assert manifest["a.pdf"] == PaperInfo(year=2024, own_paper=True)
    assert manifest["b.pdf"] == PaperInfo(year=None, own_paper=False)


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
