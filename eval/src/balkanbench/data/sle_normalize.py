"""Row normalization for the Serbian LLM Eval (SLE) dataset publish flow.

Pure functions only: takes raw upstream JSONL rows (as loaded from
``gordicaleksa/serbian-llm-eval-v1``'s per-task JSONL partials) and returns
normalized rows matching the schemas published to ``permitt/serbian-llm-eval``.
No network, no HuggingFace imports - keeps this unit-testable in isolation
from ``publish_sle_dataset.py``.

Row schemas after normalization:
    - MC tasks (arc_challenge, arc_easy, hellaswag, openbookqa):
        {example_id: str, query: str, choices: list[str], gold: int}
    - piqa (upstream field is "goal", not "query" - kept as-is to match the
      already-published ``piqa`` task config's ``inputs.fields``):
        {example_id: str, goal: str, choices: list[str], gold: int}
    - winogrande: {example_id: str, sentence: str, option1: str, option2: str, answer: str}
    - boolq: {example_id: str, question: str, passage: str, label: int}
    - nq_open: {example_id: str, question: str, answer: list[str]}
    - triviaqa: {example_id: str, question: str, answer_value: str, answer_aliases: list[str]}

``example_id`` policy (applied in this order):
    1. If the raw row has an ``id`` field, use it (str-cast).
    2. Else, for ``boolq``, use ``f"boolq-{split}-{row['idx']}"``.
    3. Else, use ``f"{task}-{split}-{index}"`` where ``index`` is the row's
       0-based position in the input list.
"""

from __future__ import annotations

# Raw fields required on every row for a given task. Used purely for
# validation (missing fields -> ValueError); this is not the output schema.
TASK_FIELDS: dict[str, tuple[str, ...]] = {
    "arc_challenge": ("query", "choices", "gold"),
    "arc_easy": ("query", "choices", "gold"),
    "hellaswag": ("query", "choices", "gold"),
    "openbookqa": ("query", "choices", "gold"),
    "piqa": ("goal", "choices", "gold"),
    "boolq": ("question", "passage", "label", "idx"),
    "winogrande": ("sentence", "option1", "option2", "answer"),
    "nq_open": ("question", "answer"),
    "triviaqa": ("question", "answer"),
}

# Tasks whose output schema is {example_id, <primary_field>, choices, gold}.
# The primary field name varies: most use "query", piqa uses "goal" because
# that's the field name upstream and it's what the piqa task config declares.
_MC_PRIMARY_FIELD: dict[str, str] = {
    "arc_challenge": "query",
    "arc_easy": "query",
    "hellaswag": "query",
    "openbookqa": "query",
    "piqa": "goal",
}


def _example_id(task: str, split: str, row: dict, index: int) -> str:
    if "id" in row:
        return str(row["id"])
    if task == "boolq":
        return f"boolq-{split}-{row['idx']}"
    return f"{task}-{split}-{index}"


def _require_fields(task: str, row: dict) -> None:
    missing = [f for f in TASK_FIELDS[task] if f not in row]
    if missing:
        raise ValueError(f"row missing required field(s) {missing} for task {task!r}: {row!r}")


def _normalize_mc(task: str, split: str, rows: list[dict]) -> list[dict]:
    field = _MC_PRIMARY_FIELD[task]
    out = []
    for i, row in enumerate(rows):
        _require_fields(task, row)
        out.append(
            {
                "example_id": _example_id(task, split, row, i),
                field: row[field],
                "choices": row["choices"],
                "gold": row["gold"],
            }
        )
    return out


def _normalize_boolq(task: str, split: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        _require_fields(task, row)
        out.append(
            {
                "example_id": _example_id(task, split, row, i),
                "question": row["question"],
                "passage": row["passage"],
                "label": row["label"],
            }
        )
    return out


def _normalize_winogrande(task: str, split: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        _require_fields(task, row)
        out.append(
            {
                "example_id": _example_id(task, split, row, i),
                "sentence": row["sentence"],
                "option1": row["option1"],
                "option2": row["option2"],
                "answer": row["answer"],
            }
        )
    return out


def _normalize_nq_open(task: str, split: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        _require_fields(task, row)
        out.append(
            {
                "example_id": _example_id(task, split, row, i),
                "question": row["question"],
                "answer": row["answer"],
            }
        )
    return out


def _normalize_triviaqa(task: str, split: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        _require_fields(task, row)
        answer = row["answer"]
        if not isinstance(answer, dict) or "value" not in answer or "aliases" not in answer:
            raise ValueError(f"triviaqa row missing answer.value/answer.aliases: {row!r}")
        out.append(
            {
                "example_id": _example_id(task, split, row, i),
                "question": row["question"],
                "answer_value": answer["value"],
                "answer_aliases": list(answer["aliases"]),
            }
        )
    return out


_DISPATCH = {
    "arc_challenge": _normalize_mc,
    "arc_easy": _normalize_mc,
    "hellaswag": _normalize_mc,
    "openbookqa": _normalize_mc,
    "piqa": _normalize_mc,
    "boolq": _normalize_boolq,
    "winogrande": _normalize_winogrande,
    "nq_open": _normalize_nq_open,
    "triviaqa": _normalize_triviaqa,
}


def normalize_rows(task: str, split: str, rows: list[dict]) -> list[dict]:
    """Normalize raw upstream ``rows`` for ``task``/``split`` to the published schema.

    Raises ``ValueError`` if ``task`` is not one of the 9 known SLE tasks, or
    if any row is missing a field required for that task.
    """
    handler = _DISPATCH.get(task)
    if handler is None:
        raise ValueError(f"unknown task: {task!r}; expected one of {sorted(_DISPATCH)}")
    return handler(task, split, rows)
