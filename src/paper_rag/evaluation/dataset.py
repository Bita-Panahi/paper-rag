"""The evaluation set: questions YOU wrote, with the facts a correct answer must contain."""

import csv
from dataclasses import dataclass
from pathlib import Path

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    question: str
    expected_files: tuple[str, ...]  # papers that contain the answer (empty = not checked)
    must_contain: tuple[str, ...]  # short facts a correct answer must include
    answerable: bool  # False = the papers do NOT contain the answer; the system should refuse


def _split(cell: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in cell.split(";") if part.strip())


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Read eval/questions.csv: question, expected_file, must_contain, answerable."""
    questions: list[EvalQuestion] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        # skipinitialspace: tolerate `a, "b", c` (a space after each comma)
        reader = csv.DictReader(f, skipinitialspace=True)
        names = [(n or "").strip().lower() for n in reader.fieldnames or []]
        missing = {"question", "answerable"} - set(names)
        if missing:
            raise DatasetError(f"{path}: missing column(s): {', '.join(sorted(missing))}")
        reader.fieldnames = names
        for line_no, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            flag = (row.get("answerable") or "yes").strip().lower()
            if flag not in _YES | _NO:
                raise DatasetError(
                    f"line {line_no}: answerable must be exactly yes or no, got '{flag}'"
                )
            answerable = flag in _YES
            facts = _split(row.get("must_contain") or "")
            if answerable and not facts:
                raise DatasetError(f"line {line_no}: an answerable question needs must_contain")
            questions.append(
                EvalQuestion(
                    id=len(questions) + 1,
                    question=question,
                    expected_files=_split(row.get("expected_file") or ""),
                    must_contain=facts,
                    answerable=answerable,
                )
            )
    if not questions:
        raise DatasetError(f"{path}: no questions found")
    return questions
