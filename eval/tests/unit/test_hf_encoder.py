"""Tests for the HF encoder wrapper (no real model download)."""

from __future__ import annotations

from typing import Any

import pytest

from balkanbench.models.hf_encoder import HFEncoder


class _FakeModel:
    def __init__(self, source: str, **kwargs: Any) -> None:
        self.source = source
        self.kwargs = kwargs


class _FakeTokenizer:
    def __init__(self, source: str, **kwargs: Any) -> None:
        self.source = source
        self.kwargs = kwargs


def _classification_task_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "boolq",
            "splits": {"public": ["train"], "labeled_public": ["train"]},
        },
        "inputs": {"fields": ["question", "passage"], "id_field": "example_id"},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "boolq_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def _multiple_choice_task_cfg() -> dict:
    cfg = _classification_task_cfg()
    cfg["task"] = "copa"
    cfg["task_type"] = "multiple_choice"
    cfg["num_choices"] = 2
    cfg["inputs"] = {
        "fields": ["premise", "choice1", "choice2", "question"],
        "id_field": "example_id",
    }
    return cfg


def _bertic_model_cfg() -> dict:
    return {
        "name": "bertic",
        "hf_repo": "classla/bcms-bertic",
        "family": "electra",
        "params_hint": "110M",
        "tier": "official",
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 10,
            "fp16": True,
        },
        "seeds": [42, 43, 44, 45, 46],
    }


def test_build_classification_model(monkeypatch) -> None:
    captured = {}

    def fake_auto_model(path: str, **kwargs: Any):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _FakeModel(path, **kwargs)

    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoModelForSequenceClassification.from_pretrained",
        fake_auto_model,
    )
    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoTokenizer.from_pretrained",
        lambda path, **kwargs: _FakeTokenizer(path, **kwargs),
    )

    enc = HFEncoder.build(
        model_cfg=_bertic_model_cfg(),
        task_cfg=_classification_task_cfg(),
    )
    assert isinstance(enc.model, _FakeModel)
    assert enc.model.source == "classla/bcms-bertic"
    assert enc.model.kwargs["num_labels"] == 2
    assert isinstance(enc.tokenizer, _FakeTokenizer)


def test_build_multiple_choice_model(monkeypatch) -> None:
    captured = {}

    def fake_auto_mc(path: str, **kwargs: Any):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _FakeModel(path, **kwargs)

    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoModelForMultipleChoice.from_pretrained",
        fake_auto_mc,
    )
    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoTokenizer.from_pretrained",
        lambda path, **kwargs: _FakeTokenizer(path, **kwargs),
    )

    enc = HFEncoder.build(
        model_cfg=_bertic_model_cfg(),
        task_cfg=_multiple_choice_task_cfg(),
    )
    assert captured["path"] == "classla/bcms-bertic"
    assert isinstance(enc.model, _FakeModel)


def test_task_overrides_merge_into_training_args(monkeypatch) -> None:
    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoModelForSequenceClassification.from_pretrained",
        lambda path, **kwargs: _FakeModel(path, **kwargs),
    )
    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoTokenizer.from_pretrained",
        lambda path, **kwargs: _FakeTokenizer(path, **kwargs),
    )

    model_cfg = _bertic_model_cfg()
    model_cfg["task_overrides"] = {"superglue.boolq": {"num_epochs": 30, "learning_rate": 1e-5}}
    enc = HFEncoder.build(model_cfg=model_cfg, task_cfg=_classification_task_cfg())
    # The merged training args should reflect the override
    assert enc.training_args["num_epochs"] == 30
    assert enc.training_args["learning_rate"] == 1e-5
    assert enc.training_args["batch_size"] == 16  # from task config (unchanged)


def test_rejects_unknown_task_type(monkeypatch) -> None:
    monkeypatch.setattr(
        "balkanbench.models.hf_encoder.AutoTokenizer.from_pretrained",
        lambda path, **kwargs: _FakeTokenizer(path, **kwargs),
    )
    cfg = _classification_task_cfg()
    cfg["task_type"] = "regression_weirdness"
    with pytest.raises(ValueError, match="task_type"):
        HFEncoder.build(model_cfg=_bertic_model_cfg(), task_cfg=cfg)
