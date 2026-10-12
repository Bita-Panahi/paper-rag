import pytest
from helpers import build

from paper_rag.ingest.manifest import PaperInfo
from paper_rag.retrieval.scope import PaperResolver
from paper_rag.store import SearchFilters

PAPERS = {
    "CIC_2024_7_Bita--Panahi.pdf": PaperInfo(
        title="Estimating Spectral BRDF Parameters Using Handheld Devices", aliases=("CIC 2024",)
    ),
    "Optics_Express.pdf": PaperInfo(aliases=("Optics Express paper",)),
}
CIC, OPTICS = "CIC_2024_7_Bita--Panahi.pdf", "Optics_Express.pdf"


@pytest.fixture
def resolver():
    return PaperResolver(PAPERS)


def test_alias_is_found_ignoring_case_and_punctuation(resolver):
    assert resolver.resolve("Which tools are used in the cic-2024 paper?") == (CIC,)


def test_file_name_stem_works_without_any_alias(resolver):
    assert resolver.resolve("What cost function is used in Optics Express?") == (OPTICS,)


def test_title_works_without_any_alias(resolver):
    q = "Summarise Estimating Spectral BRDF Parameters Using Handheld Devices"
    assert resolver.resolve(q) == (CIC,)


def test_question_naming_two_papers_returns_both(resolver):
    assert set(resolver.resolve("Compare the CIC 2024 paper with the Optics Express paper")) == {
        CIC,
        OPTICS,
    }


def test_no_paper_named_means_no_scope(resolver):
    assert resolver.resolve("Which papers use neural networks?") == ()


def test_only_whole_words_match():
    resolver = PaperResolver({"a.pdf": PaperInfo(aliases=("CIC",))})
    assert resolver.resolve("this is a specific question") == ()
    assert resolver.resolve("what does CIC say") == ("a.pdf",)


def test_very_short_aliases_are_ignored():
    assert PaperResolver({"a.pdf": PaperInfo(aliases=("ab",))}).resolve("ab") == ()


def scoped_service(**kwargs):
    resolver = PaperResolver(
        {"coating.pdf": PaperInfo(aliases=("the coating paper",)), "neural.pdf": PaperInfo()}
    )
    return build(resolver=resolver, **kwargs)[0]


def test_named_paper_limits_the_search_to_that_paper():
    service = scoped_service()
    # "network" matches neural.pdf best, but the question names the coating paper
    result = service.ask("what does the coating paper say about a latent network?")
    assert result.scoped_files == ["coating.pdf"]
    assert {s.file_name for s in result.sources} <= {"coating.pdf"}


def test_without_a_named_paper_all_papers_are_searched():
    result = scoped_service().ask("latent brdf representation network pigment flakes")
    assert result.scoped_files == []
    assert {s.file_name for s in result.sources} == {"coating.pdf", "neural.pdf"}


def test_explicit_file_filter_wins_over_the_question():
    service = scoped_service()
    result = service.ask(
        "what does the coating paper say?", filters=SearchFilters(file_name="neural.pdf")
    )
    assert result.scoped_files == []
    assert {s.file_name for s in result.sources} <= {"neural.pdf"}


def test_scoping_can_be_switched_off():
    result = scoped_service(scope=False).ask("what does the coating paper say about a network?")
    assert result.scoped_files == []
    assert {s.file_name for s in result.sources} == {"coating.pdf", "neural.pdf"}
