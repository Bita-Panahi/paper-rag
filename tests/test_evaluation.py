import json

import pytest
from helpers import build

from paper_rag.evaluation.dataset import DatasetError, load_questions
from paper_rag.evaluation.metrics import fact_recall, first_rank, normalize, percentile
from paper_rag.evaluation.runner import run_evaluation


def write_csv(tmp_path, body):
    path = tmp_path / "q.csv"
    path.write_text("question,expected_file,must_contain,answerable\n" + body, encoding="utf-8")
    return path


def test_normalize_ignores_case_spaces_and_symbols():
    assert normalize("Canon\nSRA (RA-532H)") == "canonsrara532h"


def test_fact_recall_counts_found_facts():
    text = "We used the X-Rite MA-T12 and a Canon  SRA device."
    assert fact_recall(text, ["MA-T12", "canon sra"]) == 1.0
    assert fact_recall(text, ["MA-T12", "Spectralon"]) == 0.5
    assert fact_recall(text, []) == 1.0


def test_first_rank_is_one_based_or_none():
    assert first_rank(["a.pdf", "b.pdf", "b.pdf"], ["b.pdf"]) == 2
    assert first_rank(["a.pdf"], ["z.pdf"]) is None


def test_percentile_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([], 50) == 0.0


def test_load_questions_parses_columns(tmp_path):
    path = write_csv(tmp_path, "q1,a.pdf;b.pdf,fact one; fact two,yes\nq2,,,no\n")
    first, second = load_questions(path)
    assert first.expected_files == ("a.pdf", "b.pdf")
    assert first.must_contain == ("fact one", "fact two") and first.answerable
    assert not second.answerable and second.must_contain == ()


def test_answerable_question_without_facts_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="must_contain"):
        load_questions(write_csv(tmp_path, "q1,a.pdf,,yes\n"))


def test_bad_answerable_flag_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="answerable"):
        load_questions(write_csv(tmp_path, "q1,a.pdf,x,maybe\n"))


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="no questions"):
        load_questions(write_csv(tmp_path, ""))


def questions(tmp_path):
    return load_questions(
        write_csv(
            tmp_path,
            "how do coating pigments shift colour at angles,coating.pdf,pigment flakes,yes\n"
            "what is a latent brdf representation,wrong.pdf,latent,yes\n"
            "completely unrelated quantum zebra,,,no\n",
        )
    )


def test_retrieval_only_evaluation(tmp_path):
    service, llm = build(min_score=0.3)
    results, summary = run_evaluation(service, questions(tmp_path), with_answers=False)

    assert [r.passed for r in results] == [True, False, True]  # 2nd expects a paper that is absent
    assert results[0].file_rank == 1 and results[1].file_rank is None
    assert llm.calls == []  # retrieval-only mode never calls the LLM
    assert summary["paper_hit_rate"] == 0.5 and summary["refusal_accuracy"] == 1.0


def test_full_evaluation_checks_answers_and_refusals(tmp_path):
    service, llm = build(reply="Pigment flakes shift colour [1].", min_score=0.3)
    results, summary = run_evaluation(service, questions(tmp_path), with_answers=True)

    assert results[0].answer_recall == 1.0 and results[0].passed
    assert results[2].passed and results[2].answered is False  # refused without paying the LLM
    assert len(llm.calls) == 2  # the unanswerable question never reached the LLM
    assert summary["questions"] == 3 and summary["total_cost_nok"] > 0
    json.dumps(summary)  # the summary must be JSON-serialisable for the report file


def test_failing_question_is_recorded_not_raised(tmp_path, monkeypatch):
    service, _ = build()
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: 1 / 0)
    results, summary = run_evaluation(service, questions(tmp_path), with_answers=False)
    assert all(r.error and not r.passed for r in results) and summary["errors"] == 3


def test_spaces_after_commas_and_header_spaces_are_tolerated(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(
        "question, expected_file, must_contain, answerable\n"
        '"Which tools?", "a.pdf", "MA-T12; Canon SRA", yes\n',
        encoding="utf-8",
    )
    (q,) = load_questions(path)
    assert q.expected_files == ("a.pdf",) and q.must_contain == ("MA-T12", "Canon SRA")


def test_missing_column_is_reported(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("question,expected_file\nq,a.pdf\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="answerable"):
        load_questions(path)


def test_context_recall_ignores_text_from_other_papers(tmp_path):
    # "latent" exists in neural.pdf, but the question expects coating.pdf: it must not count
    service, _ = build(min_score=0.3)
    q = load_questions(
        write_csv(tmp_path, "what is a latent brdf representation,coating.pdf,latent,yes\n")
    )
    (result,), _ = run_evaluation(service, q, with_answers=False)
    assert result.context_recall == 0.0 and result.missing_facts == ["latent"]
    assert not result.passed


def test_correct_words_cited_from_the_wrong_paper_do_not_pass(tmp_path):
    service, _ = build(reply="Latent representation [1].", min_score=0.3)
    q = load_questions(
        write_csv(tmp_path, "what is a latent brdf representation,coating.pdf,latent,yes\n")
    )
    (result,), summary = run_evaluation(service, q, with_answers=True)
    assert result.answer_recall == 1.0 and result.cited_expected is False
    assert not result.passed and summary["citation_hit_rate"] == 0.0


def test_missing_facts_lists_what_was_not_found():
    from paper_rag.evaluation.metrics import missing_facts

    assert missing_facts("We used MA-T12.", ["MA-T12", "Canon SRA"]) == ["Canon SRA"]


def test_refusal_with_extra_made_up_text_does_not_pass(tmp_path):
    from paper_rag.retrieval.prompts import NOT_FOUND_MESSAGE

    rows = "pigment flakes shift colour,,,no\n"
    mixed, _ = build(reply=f"They shift colour. {NOT_FOUND_MESSAGE}", min_score=0.3)
    clean, _ = build(reply=NOT_FOUND_MESSAGE, min_score=0.3)
    questions = load_questions(write_csv(tmp_path, rows))
    assert not run_evaluation(mixed, questions)[0][0].passed
    assert run_evaluation(clean, questions)[0][0].passed
