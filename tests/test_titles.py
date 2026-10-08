import pytest

from paper_rag.ingest.titles import resolve_title


def test_good_metadata_title_is_used():
    title = "Neural BRDF Representation and Importance Sampling"
    assert resolve_title(title, "x.pdf") == title


@pytest.mark.parametrize(
    "junk",
    [
        None,
        "",
        "   ",
        "Microsoft Word - measuring and specifying special effects colours.docx",
        "Makale Bilgileri/ Article info Geliş / Received: **.**.20**",
        "Untitled",
    ],
)
def test_junk_titles_fall_back_to_file_name(junk):
    assert resolve_title(junk, "Hypernetworks_for_Generalizable BRDF.pdf") == (
        "Hypernetworks for Generalizable BRDF"
    )
