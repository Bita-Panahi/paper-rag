"""Run the evaluation:  python -m paper_rag.evaluation --label baseline

--no-answer  retrieval only: free apart from tiny embedding costs, no LLM calls.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from paper_rag.config import get_settings
from paper_rag.evaluation.dataset import DatasetError, load_questions
from paper_rag.evaluation.runner import result_rows, run_evaluation
from paper_rag.retrieval.service import RagService


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument("--questions", default="eval/questions.csv")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--no-scope",
        action="store_true",
        help="do not limit the search to papers named in the question",
    )
    parser.add_argument("--no-answer", action="store_true", help="retrieval only, no LLM calls")
    parser.add_argument("--label", default="run", help="short name for this experiment")
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions)
    except (DatasetError, OSError) as exc:
        print(f"Cannot read questions: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    overrides = {k: v for k, v in {"min_score": args.min_score}.items() if v is not None}
    if args.no_scope:
        overrides["scope_to_named_papers"] = False
    service = RagService.from_settings(settings.model_copy(update=overrides))
    with_answers = not args.no_answer

    results, summary = run_evaluation(
        service, questions, top_k=args.top_k, with_answers=with_answers
    )

    for r in results:
        rank = "-" if r.file_rank is None else str(r.file_rank)
        recall = "-" if r.context_recall is None else f"{r.context_recall:.2f}"
        mark = "PASS" if r.passed else "FAIL"
        note = "  [scoped]" if r.scoped_files else ""
        note += f"  ERROR {r.error}" if r.error else ""
        if not r.passed and r.missing_facts:
            note += f"  [missing: {'; '.join(r.missing_facts)}]"
        print(
            f"{mark} #{r.id:<3} paper@{rank:<2} ctx={recall:<4} {r.total_ms:6.0f} ms  "
            f"{r.question[:70]}{note}"
        )
    print()
    for key, value in summary.items():
        print(f"{key:>22}: {value}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"{stamp}_{args.label}.json"
    config = {
        "top_k": args.top_k or settings.top_k,
        "min_score": overrides.get("min_score", settings.min_score),
        "scope_to_named_papers": overrides.get(
            "scope_to_named_papers", settings.scope_to_named_papers
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "with_answers": with_answers,
    }
    out_file.write_text(
        json.dumps(
            {
                "label": args.label,
                "config": config,
                "summary": summary,
                "results": result_rows(results),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
