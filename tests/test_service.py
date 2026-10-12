import pytest
from helpers import build

from paper_rag.retrieval.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from paper_rag.store import SearchFilters


def test_answer_has_sources_citations_and_usage():
    service, llm = build()
    result = service.ask("how do coating pigments shift colour at angles?")

    assert result.answered and result.answer == "Coatings shift colour [1]."
    assert result.sources and result.sources[0].number == 1
    assert result.sources[0].cited and not any(s.cited for s in result.sources[1:])
    assert len(llm.calls) == 1
    u = result.usage
    assert u.input_tokens > 0 and u.cost_nok > 0 and u.total_ms >= u.retrieval_ms >= 0


def test_prompt_contains_numbered_excerpts_with_file_and_page():
    service, llm = build()
    service.ask("coating pigments")
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] " in user and ".pdf, p. " in user and "Question: coating pigments" in user


def test_metadata_filter_limits_the_sources():
    service, _ = build()
    result = service.ask("brdf network coating", filters=SearchFilters(own_paper=False))
    assert {s.file_name for s in result.sources} == {"neural.pdf"}


def test_no_relevant_hits_skips_the_llm_call():
    service, llm = build(min_score=0.99)
    result = service.ask("something completely unrelated xyz")
    assert result.answer == NOT_FOUND_MESSAGE and not result.answered and result.sources == []
    assert llm.calls == []  # no tokens paid
    assert result.usage.input_tokens == 0 and result.usage.generation_ms == 0


def test_model_saying_not_found_is_marked_unanswered():
    service, _ = build(reply=NOT_FOUND_MESSAGE)
    assert service.ask("coating pigments").answered is False


def test_build_user_prompt_formats_page_ranges():
    service, _ = build()
    hits, _ = service.retrieve("coating")
    hit = hits[0]
    from dataclasses import replace

    multi = replace(hit, page_start=2, page_end=3)
    assert "pp. 2-3" in build_user_prompt("q", [multi])


@pytest.mark.parametrize("top_k", [1, 2])
def test_top_k_limits_sources(top_k):
    service, _ = build()
    assert len(service.ask("coating neural brdf pigment", top_k=top_k).sources) <= top_k


def test_answer_that_adds_the_refusal_sentence_counts_as_unanswered():
    mixed = f"Some guess about the paper. {NOT_FOUND_MESSAGE}"
    service, _ = build(reply=mixed)
    assert service.ask("coating pigments").answered is False
