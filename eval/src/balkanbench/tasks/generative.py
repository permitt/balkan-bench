"""Generative task classes: prompt construction + parsing.

Covers two ``task_type`` values used by the SLE track:

- ``multiple_choice_loglikelihood`` (:class:`MultipleChoiceLoglikelihoodTask`):
  ARC / HellaSwag / OpenBookQA / PIQA (plain multiple choice), plus two
  variants with bespoke prompt construction:
  - ``winogrande_partial`` (WinoGrande partial-context scoring)
  - ``boolq_da_ne`` (BoolQ scored via " da" / " ne" continuations)
- ``generative_qa`` (:class:`GenerativeQATask`): NQ-Open / TriviaQA, scored
  by greedy generation + exact match.

Both task types are encoder-only-incompatible: they are consumed by a
generation/loglikelihood harness, not ``Trainer``/``AutoModelForSequenceClassification``,
so ``preprocess``/``decode`` (the encoder-only ``Task`` abstract methods) are
not meaningful here and raise ``NotImplementedError("generative task")``.

Parity with the reference harness (gordicaleksa/serbian-llm-eval @
serb_eval_run, i.e. lm_eval v0.3.0) is pinned as follows - see
``.superpowers/sdd/task-6-report.md`` for the full quotes:

- ``lm_eval/base.py`` ``MultipleChoiceTask``: continuation is ``" " + choice``;
  ``process_results`` normalizes acc_norm by
  ``completion_len = np.array([float(len(i)) for i in doc["choices"]])`` -
  plain Python ``len()`` on the choice string, i.e. **character count**, not
  UTF-8 byte length. Serbian diacritics (č, ć, š, ž, đ) are single characters
  but 2 bytes in UTF-8, so this distinction is load-bearing.
- ``lm_eval/tasks/winogrande.py``: ``partial_context`` = ``sentence[:pronoun_loc] +
  option`` (sentence truncated at the ``"_"`` index, option appended with no
  extra separator); ``partial_target`` = ``" " + sentence[pronoun_loc + 1:].strip()``;
  ``answer_to_num = {"1": 0, "2": 1}``.
- ``lm_eval/tasks/superglue.py`` ``BoolQ`` (Serbian branch): prompt is
  ``f"{passage}\\nPitanje: {question}?\\nOdgovor:"``; ``construct_requests``
  builds ``yes`` (" da") **then** ``no`` (" ne") and returns ``(ll_yes, ll_no)``
  - i.e. da is request index 0, ne is index 1. This is the *opposite* order
  from the task brief's draft test; the fork wins, so ``gold_index`` for
  ``label == 1`` (yes/da) is **0**, not 1.
- ``lm_eval/tasks/nqopen.py`` / ``triviaqa.py``: prompt is
  ``f"Pitanje: {question}\\nOdgovor:"`` for both tasks (Serbian branch).

The API-protocol prompts (``api_prompt`` / ``parse_api_response``) are our own
protocol with no fork equivalent; they exist only on the multiple-choice side
(the generative_qa API reformulation is plain generation using ``qa_prompt``,
so no letter/da-ne parsing is needed there).
"""

from __future__ import annotations

import re
from typing import Any

from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task

_LETTERS = "abcdefghij"  # A-J / a-j, per the brief's parsing regex.
_LETTER_RE = re.compile(r"^[\s\(\[]*([A-Ja-j])\b")
_DA_NE_RE = re.compile(r"^[\s\"']*(da|ne)\b", re.IGNORECASE)

_WINOGRANDE_VARIANT = "winogrande_partial"
_BOOLQ_VARIANT = "boolq_da_ne"


@register_task("multiple_choice_loglikelihood")
class MultipleChoiceLoglikelihoodTask(Task):
    """Loglikelihood-scored multiple choice: ARC/HellaSwag/OpenBookQA/PIQA,
    plus the WinoGrande partial-context and BoolQ da/ne variants.
    """

    task_type = "multiple_choice_loglikelihood"

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        self._variant: str | None = cfg.get("variant")
        fields = cfg["inputs"]["fields"]
        # arc/hellaswag/openbookqa use "query", piqa uses "goal"; winogrande
        # and boolq variants build their own prompts and ignore this field.
        self._prompt_field: str = fields[0]

    # ------------------------------------------------------------------
    # Encoder-only Task ABC methods: not applicable to a generative task.
    # ------------------------------------------------------------------

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        raise NotImplementedError("generative task")

    def decode(self, logits: Any) -> Any:
        raise NotImplementedError("generative task")

    # ------------------------------------------------------------------
    # Loglikelihood scoring protocol
    # ------------------------------------------------------------------

    def loglikelihood_requests(self, ex: dict[str, Any]) -> list[tuple[str, str]]:
        if self._variant == _WINOGRANDE_VARIANT:
            return self._winogrande_requests(ex)
        if self._variant == _BOOLQ_VARIANT:
            return self._boolq_requests(ex)
        return self._mc_requests(ex)

    def gold_index(self, ex: dict[str, Any]) -> int:
        if self._variant == _WINOGRANDE_VARIANT:
            return {"1": 0, "2": 1}[ex["answer"]]
        if self._variant == _BOOLQ_VARIANT:
            # Fork request order is [da, ne]; label 1 (yes) -> da -> index 0.
            return 0 if ex["label"] == 1 else 1
        return int(ex["gold"])

    def continuation_lengths(self, ex: dict[str, Any]) -> list[int]:
        if self._variant in (_WINOGRANDE_VARIANT, _BOOLQ_VARIANT):
            # acc only for these two variants; lengths are unused by the metric.
            return [1] * len(self.loglikelihood_requests(ex))
        # acc_norm normalization: len(choice) CHARACTERS (see module docstring).
        return [len(choice) for choice in ex["choices"]]

    def _mc_requests(self, ex: dict[str, Any]) -> list[tuple[str, str]]:
        query = ex[self._prompt_field]
        return [(query, " " + choice) for choice in ex["choices"]]

    def _winogrande_requests(self, ex: dict[str, Any]) -> list[tuple[str, str]]:
        sentence = ex["sentence"]
        pronoun_loc = sentence.index("_")
        target = " " + sentence[pronoun_loc + 1 :].strip()
        return [
            (sentence[:pronoun_loc] + ex["option1"], target),
            (sentence[:pronoun_loc] + ex["option2"], target),
        ]

    def _boolq_requests(self, ex: dict[str, Any]) -> list[tuple[str, str]]:
        ctx = f"{ex['passage']}\nPitanje: {ex['question']}?\nOdgovor:"
        return [(ctx, " da"), (ctx, " ne")]

    # ------------------------------------------------------------------
    # API-protocol (multiple-choice reformulation): our own protocol.
    # ------------------------------------------------------------------

    def api_prompt(self, ex: dict[str, Any]) -> str:
        if self._variant == _WINOGRANDE_VARIANT:
            return self._winogrande_api_prompt(ex)
        if self._variant == _BOOLQ_VARIANT:
            return self._boolq_api_prompt(ex)
        return self._mc_api_prompt(ex)

    def parse_api_response(self, text: str, ex: dict[str, Any]) -> int | None:
        if self._variant == _BOOLQ_VARIANT:
            return self._parse_da_ne(text)
        return self._parse_letter(text, len(ex["choices"]))

    def _mc_api_prompt(self, ex: dict[str, Any]) -> str:
        query = ex[self._prompt_field]
        lines = [query]
        for i, choice in enumerate(ex["choices"]):
            lines.append(f"{_LETTERS[i].upper()}. {choice}")
        return "\n".join(lines) + "\n\nOdgovori samo slovom tačnog odgovora."

    def _winogrande_api_prompt(self, ex: dict[str, Any]) -> str:
        sentence = ex["sentence"]
        option1_sentence = sentence.replace("_", ex["option1"])
        option2_sentence = sentence.replace("_", ex["option2"])
        return (
            "Koja rečenica je smislenija?\n"
            f"A. {option1_sentence}\n"
            f"B. {option2_sentence}\n\n"
            "Odgovori samo slovom A ili B."
        )

    def _boolq_api_prompt(self, ex: dict[str, Any]) -> str:
        return f'{ex["passage"]}\nPitanje: {ex["question"]}?\nOdgovori samo sa "da" ili "ne".'

    @staticmethod
    def _parse_letter(text: str, num_choices: int) -> int | None:
        match = _LETTER_RE.match(text.strip())
        if match is None:
            return None
        idx = _LETTERS.index(match.group(1).lower())
        if idx >= num_choices:
            return None
        return idx

    @staticmethod
    def _parse_da_ne(text: str) -> int | None:
        match = _DA_NE_RE.match(text.strip())
        if match is None:
            return None
        return 1 if match.group(1).lower() == "da" else 0


@register_task("generative_qa")
class GenerativeQATask(Task):
    """Greedy-generation QA: NQ-Open / TriviaQA.

    Both use the same Serbian prompt template as the fork's
    ``nqopen.py`` / ``triviaqa.py``: ``"Pitanje: {question}\\nOdgovor:"``.
    """

    task_type = "generative_qa"

    # ------------------------------------------------------------------
    # Encoder-only Task ABC methods: not applicable to a generative task.
    # ------------------------------------------------------------------

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        raise NotImplementedError("generative task")

    def decode(self, logits: Any) -> Any:
        raise NotImplementedError("generative task")

    # ------------------------------------------------------------------
    # QA prompt + references
    # ------------------------------------------------------------------

    def qa_prompt(self, ex: dict[str, Any]) -> str:
        return f"Pitanje: {ex['question']}\nOdgovor:"

    def qa_references(self, ex: dict[str, Any]) -> list[str]:
        if "answer_value" in ex:  # triviaqa: {answer_value, answer_aliases}
            return [ex["answer_value"], *ex.get("answer_aliases", [])]
        return list(ex["answer"])  # nq_open: answer is list[str]

    def fewshot_example_text(self, ex: dict[str, Any]) -> str:
        refs = self.qa_references(ex)
        return f"{self.qa_prompt(ex)} {refs[0]}"


__all__ = ["MultipleChoiceLoglikelihoodTask", "GenerativeQATask"]
