"""Run every evaluation question through the system and score it."""

import logging
import time
from dataclasses import asdict, dataclass, field

from paper_rag.evaluation.dataset import EvalQuestion
from paper_rag.evaluation.metrics import fact_recall, first_rank, missing_facts, percentile
from paper_rag.llm.embeddings import embedding_cost_nok
from paper_rag.retrieval.prompts import NOT_FOUND_MESSAGE
from paper_rag.retrieval.service import RagService

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    id: int
    question: str
    answerable: bool
    expected_files: list[str]
    scoped_files: list[str] = field(default_factory=list)  # search limited to these papers
    retrieved_files: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # file, page, score of each hit
    missing_facts: list[str] = field(default_factory=list)  # facts not found (see `passed`)
    cited_expected: bool | None = None  # did the answer cite an expected paper?
    file_rank: int | None = None  # where the right paper first appears (None = not retrieved)
    context_recall: float | None = None  # facts found in the retrieved text
    answer_recall: float | None = None  # facts found in the generated answer
    answered: bool | None = None  # did the system answer (True) or refuse (False)?
    answer: str = ""
    passed: bool = False
    cost_nok: float = 0.0
    total_ms: float = 0.0
    error: str = ""


def evaluate_question(
    service: RagService, q: EvalQuestion, *, top_k: int | None, with_answers: bool
) -> QuestionResult:
    result = QuestionResult(q.id, q.question, q.answerable, list(q.expected_files))
    started = time.perf_counter()
    try:
        result.scoped_files = service.scope(q.question, None)[1]
        hits, embedding_tokens = service.retrieve(q.question, top_k)
        result.retrieved_files = [h.file_name for h in hits]
        result.retrieved = [
            {"file": h.file_name, "page": h.page_start, "score": round(h.score, 3)} for h in hits
        ]
        result.cost_nok = embedding_cost_nok(embedding_tokens, service.settings)
        if q.expected_files:
            result.file_rank = first_rank(result.retrieved_files, q.expected_files)
        if q.answerable:
            # Only text from the expected paper(s) counts: the same fact in another paper's
            # chunk does not prove the right chunk was found.
            evidence = [h for h in hits if h.file_name in q.expected_files] or (
                [] if q.expected_files else hits
            )
            context = " ".join(h.text for h in evidence)
            result.context_recall = fact_recall(context, q.must_contain)
            result.missing_facts = missing_facts(context, q.must_contain)

        if with_answers:
            answer = service.ask(q.question, top_k)
            result.answer, result.answered = answer.answer, answer.answered
            result.cost_nok = answer.usage.cost_nok
            if q.answerable:
                result.answer_recall = fact_recall(answer.answer, q.must_contain)
                result.missing_facts = missing_facts(answer.answer, q.must_contain)
                if q.expected_files:
                    result.cited_expected = any(
                        s.cited and s.file_name in q.expected_files for s in answer.sources
                    )
                result.passed = bool(
                    answer.answered
                    and result.answer_recall == 1.0
                    and result.cited_expected is not False
                )
            else:
                # A clean refusal is exactly the not-found sentence. A made-up answer with the
                # sentence tacked on the end is still a failure.
                result.passed = answer.answer.strip() == NOT_FOUND_MESSAGE
        elif q.answerable:
            found = not q.expected_files or result.file_rank is not None
            result.passed = found and result.context_recall == 1.0
        else:
            result.passed = not hits  # nothing relevant should survive the score threshold
    except Exception as exc:  # one failing question must not stop the whole run
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("evaluation of question %d failed", q.id)
    result.total_ms = (time.perf_counter() - started) * 1000
    return result


def summarize(results: list[QuestionResult], *, with_answers: bool) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    with_expected = [r for r in answerable if r.expected_files]
    times = [r.total_ms for r in results]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    summary = {
        "questions": len(results),
        "pass_rate": mean([float(r.passed) for r in results]),
        "errors": sum(bool(r.error) for r in results),
        "paper_hit_rate": mean([float(r.file_rank is not None) for r in with_expected]),
        "mean_context_recall": mean(
            [r.context_recall for r in answerable if r.context_recall is not None]
        ),
        "refusal_accuracy": mean([float(r.passed) for r in unanswerable]),
        "total_cost_nok": round(sum(r.cost_nok for r in results), 4),
        "latency_ms_p50": round(percentile(times, 50)),
        "latency_ms_p95": round(percentile(times, 95)),
    }
    if with_answers:
        summary["mean_answer_recall"] = mean(
            [r.answer_recall for r in answerable if r.answer_recall is not None]
        )
        summary["citation_hit_rate"] = mean(
            [float(r.cited_expected) for r in answerable if r.cited_expected is not None]
        )
        summary["false_refusal_rate"] = mean([float(r.answered is False) for r in answerable])
    return summary


def run_evaluation(
    service: RagService,
    questions: list[EvalQuestion],
    *,
    top_k: int | None = None,
    with_answers: bool = True,
) -> tuple[list[QuestionResult], dict]:
    results = [
        evaluate_question(service, q, top_k=top_k, with_answers=with_answers) for q in questions
    ]
    return results, summarize(results, with_answers=with_answers)


def result_rows(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]
