"""Tests for the SLE generative task classes (prompt construction + parsing).

Parity notes (fork = gordicaleksa/serbian-llm-eval @ serb_eval_run, lm_eval v0.3.0):
- ``MultipleChoiceTask.construct_requests`` builds continuations as
  ``" " + choice`` and ``process_results`` normalizes acc_norm by
  ``float(len(choice))`` - Python string length, i.e. **characters**, not
  ``len(choice.encode("utf-8"))`` bytes. See ``lm_eval/base.py`` lines
  747-763 in the fork.
- BoolQ (``lm_eval/tasks/superglue.py``) builds requests as
  ``ll_yes, ll_no = rf.loglikelihood(ctx, " da"), rf.loglikelihood(ctx, " ne")``
  i.e. **da is requested first (index 0), ne second (index 1)** - the
  opposite order from the brief's draft test. The fork wins: this test file
  asserts ``[(ctx, " da"), (ctx, " ne")]`` and ``gold_index`` for
  ``label == 1`` (yes/da) is **0**, not 1.
"""

from __future__ import annotations

from balkanbench.cli._paths import resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.tasks import get_task_class
from balkanbench.tasks.generative import (  # noqa: F401  (ensures registration)
    GenerativeQATask,
    MultipleChoiceLoglikelihoodTask,
)


def make_task(name: str, benchmark: str = "sle", language: str = "sr"):
    cfg = load_yaml_with_schema(
        resolve_task_config(benchmark, name), schemas_root() / "task_spec.json"
    )
    task_cls = get_task_class(cfg["task_type"])
    return task_cls(cfg, language)


def test_mc_requests() -> None:
    t = make_task("arc_easy")
    ex = {"example_id": "x", "query": "Pitanje: Zašto?", "choices": ["prvi", "drugi"], "gold": 1}
    assert t.loglikelihood_requests(ex) == [
        ("Pitanje: Zašto?", " prvi"),
        ("Pitanje: Zašto?", " drugi"),
    ]
    assert t.gold_index(ex) == 1


def test_piqa_mc_requests_use_goal_field() -> None:
    t = make_task("piqa")
    ex = {"example_id": "x", "goal": "Pitanje: Zašto?", "choices": ["prvi", "drugi"], "gold": 0}
    assert t.loglikelihood_requests(ex) == [
        ("Pitanje: Zašto?", " prvi"),
        ("Pitanje: Zašto?", " drugi"),
    ]
    assert t.gold_index(ex) == 0


def test_mc_continuation_lengths_are_char_counts_not_bytes() -> None:
    # "č" is 1 char but 2 UTF-8 bytes - pins the char-vs-byte parity decision.
    t = make_task("arc_easy")
    ex = {"example_id": "x", "query": "q", "choices": ["ač", "ab"], "gold": 0}
    assert t.continuation_lengths(ex) == [2, 2]


def test_winogrande_partial_requests() -> None:
    ex = {
        "example_id": "x",
        "sentence": "Marko je dao Petru knjigu jer je _ bio velikodušan.",
        "option1": "Marko",
        "option2": "Petar",
        "answer": "1",
    }
    t = make_task("winogrande")
    reqs = t.loglikelihood_requests(ex)
    assert reqs[0] == ("Marko je dao Petru knjigu jer je Marko", " bio velikodušan.")
    assert reqs[1] == ("Marko je dao Petru knjigu jer je Petar", " bio velikodušan.")
    assert t.gold_index(ex) == 0


def test_boolq_requests() -> None:
    ex = {
        "example_id": "x",
        "question": "Da li je nebo plavo",
        "passage": "Nebo je plavo.",
        "label": 1,
    }
    t = make_task("boolq")
    assert t.loglikelihood_requests(ex) == [
        ("Nebo je plavo.\nPitanje: Da li je nebo plavo?\nOdgovor:", " da"),
        ("Nebo je plavo.\nPitanje: Da li je nebo plavo?\nOdgovor:", " ne"),
    ]
    # Fork order is [da, ne]; label 1 (yes) -> da is index 0.
    assert t.gold_index(ex) == 0


def test_qa_prompt_and_refs() -> None:
    ex = {"example_id": "x", "question": "Ko je Nikola Tesla", "answer": ["naučnik", "pronalazač"]}
    t = make_task("nq_open")
    assert t.qa_prompt(ex) == "Pitanje: Ko je Nikola Tesla\nOdgovor:"
    assert t.qa_references(ex) == ["naučnik", "pronalazač"]
    assert t.fewshot_example_text(ex) == "Pitanje: Ko je Nikola Tesla\nOdgovor: naučnik"


def test_triviaqa_refs_include_aliases() -> None:
    ex = {
        "example_id": "x",
        "question": "Q",
        "answer_value": "Tesla",
        "answer_aliases": ["N. Tesla"],
    }
    t = make_task("triviaqa")
    assert t.qa_references(ex) == ["Tesla", "N. Tesla"]


def test_api_prompt_mc() -> None:
    ex = {"example_id": "x", "query": "Pitanje: Zašto?", "choices": ["prvi", "drugi"], "gold": 0}
    t = make_task("arc_easy")
    assert t.api_prompt(ex) == (
        "Pitanje: Zašto?\nA. prvi\nB. drugi\n\nOdgovori samo slovom tačnog odgovora."
    )


def test_parse_api_response() -> None:
    t = make_task("arc_easy")
    ex = {"choices": ["a", "b", "c"]}
    assert t.parse_api_response("B", ex) == 1
    assert t.parse_api_response(" (C) zato što...", ex) == 2
    assert t.parse_api_response("b.", ex) == 1
    assert t.parse_api_response("nemam pojma", ex) is None
    assert t.parse_api_response("D", ex) is None  # out of range for 3 choices


def test_api_prompt_boolq_and_parse() -> None:
    t = make_task("boolq")
    ex = {"question": "Da li je nebo plavo", "passage": "Nebo je plavo.", "label": 1}
    assert t.api_prompt(ex) == (
        'Nebo je plavo.\nPitanje: Da li je nebo plavo?\nOdgovori samo sa "da" ili "ne".'
    )
    assert t.parse_api_response("Da", ex) == 1
    assert t.parse_api_response("ne.", ex) == 0
    assert t.parse_api_response("možda", ex) is None


def test_api_prompt_winogrande() -> None:
    ex = {"sentence": "Jer je _ bio tu.", "option1": "Marko", "option2": "Petar", "answer": "2"}
    t = make_task("winogrande")
    assert t.api_prompt(ex) == (
        "Koja rečenica je smislenija?\n"
        "A. Jer je Marko bio tu.\n"
        "B. Jer je Petar bio tu.\n\n"
        "Odgovori samo slovom A ili B."
    )


def test_winogrande_and_boolq_continuation_lengths_are_unused_ones() -> None:
    t_wino = make_task("winogrande")
    ex_wino = {
        "sentence": "Jer je _ bio tu.",
        "option1": "Marko",
        "option2": "Petar",
        "answer": "1",
    }
    assert t_wino.continuation_lengths(ex_wino) == [1, 1]

    t_boolq = make_task("boolq")
    ex_boolq = {"question": "Q", "passage": "P.", "label": 0}
    assert t_boolq.continuation_lengths(ex_boolq) == [1, 1]


def test_generic_task_abstract_methods_raise_not_implemented() -> None:
    t = make_task("arc_easy")
    try:
        t.preprocess({}, tokenizer=None)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as exc:
        assert str(exc) == "generative task"

    try:
        t.decode(None)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as exc:
        assert str(exc) == "generative task"

    qa = make_task("nq_open")
    try:
        qa.preprocess({}, tokenizer=None)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as exc:
        assert str(exc) == "generative task"
