"""``balkanbench throughput`` CLI: measure inference throughput."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.data.repo import DatasetRepoError, resolve_dataset_repo, resolve_hf_token
from balkanbench.models.hf_encoder import HFEncoder
from balkanbench.throughput import (
    ThroughputSample,
    measure_task_throughput,
    write_model_throughput_aggregate,
    write_task_throughput,
)


def __getattr__(name: str) -> Any:
    # Lazy import of datasets.load_dataset to keep `balkanbench --version` fast.
    if name == "load_dataset":
        import datasets

        return datasets.load_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


_CLASSIFICATION_LIKE = {
    "binary_classification",
    "multiclass_classification",
    "grouped_binary_classification",
    "wsc",
    "diagnostic",
}


def _task_scoped_predict_fn(task_type: str, num_choices: int) -> Any:
    """Return a predict_fn with ``task_type`` and ``num_choices`` pinned.

    Kept out of ``throughput_cmd`` so the bound values don't drift as the
    per-task loop iterates (ruff B023).
    """

    def _predict(model: Any, batch: Any, *, batch_size: int, max_seq_len: int) -> Any:
        return default_predict_fn(
            model,
            batch,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            task_type=task_type,
            num_choices=num_choices,
        )

    return _predict


def default_predict_fn(
    model: Any,
    batch: Any,
    *,
    batch_size: int,
    max_seq_len: int,
    task_type: str = "binary_classification",
    num_choices: int = 2,
) -> tuple[Any, float]:
    """Baseline predict_fn used by the CLI: runs ``model(**inputs)`` and times it.

    Input shape is chosen from ``task_type`` so the forward pass matches the
    model's actual contract:

    - classification-like (``binary_classification``, ``multiclass_classification``,
      ``grouped_binary_classification``, ``wsc``, ``diagnostic``): 2D tensor of
      shape ``(batch_size, max_seq_len)``.
    - ``multiple_choice`` (COPA / ``AutoModelForMultipleChoice``): 3D tensor of
      shape ``(batch_size, num_choices, max_seq_len)``.

    Forward-pass failures propagate. An earlier version caught every
    exception and fell back to ``input_ids.sum()``, which silently produced a
    'throughput' number that wasn't measuring a real forward pass; the
    fallback is deliberately removed.

    Tests monkeypatch this symbol directly to supply a deterministic fake.
    """
    del batch  # the real flow would index the dataset; shape-only timing here

    import numpy as np
    import torch

    if task_type == "multiple_choice":
        shape: tuple[int, ...] = (batch_size, num_choices, max_seq_len)
    elif task_type in _CLASSIFICATION_LIKE:
        shape = (batch_size, max_seq_len)
    else:
        raise ValueError(
            f"default_predict_fn has no input shape for task_type={task_type!r}; "
            f"known: {sorted(_CLASSIFICATION_LIKE | {'multiple_choice'})}"
        )

    input_ids = torch.randint(0, 1000, shape)
    attn = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = model(input_ids=input_ids, attention_mask=attn)
        elapsed = time.perf_counter() - start
    return np.zeros(batch_size, dtype=np.int64), elapsed


def _enumerate_ranked_tasks(benchmark: str, language: str) -> list[str]:
    import os

    configs_dir = Path(
        os.environ.get("BALKANBENCH_CONFIGS_DIR") or Path(__file__).resolve().parents[3] / "configs"
    )
    tasks_dir = configs_dir / "benchmarks" / benchmark / "tasks"
    out: list[str] = []
    for yaml_path in sorted(tasks_dir.glob("*.yaml")):
        cfg = load_yaml_with_schema(yaml_path, schemas_root() / "task_spec.json")
        if cfg.get("status") != "ranked":
            continue
        if language not in cfg["languages"].get("ranked", []):
            continue
        out.append(cfg["task"])
    return out


def throughput_cmd(
    model: str = typer.Option(..., "--model"),
    benchmark: str = typer.Option(..., "--benchmark"),
    language: str = typer.Option(..., "--language"),
    tasks: list[str] = typer.Option(
        None,
        "--task",
        help="Repeatable. Defaults to every ranked task for the benchmark+language.",
    ),
    hardware: str = typer.Option("NVIDIA L4 24GB", "--hardware"),
    precision: str = typer.Option("fp16", "--precision"),
    warmup_batches: int = typer.Option(2, "--warmup-batches"),
    measurement_batches: int = typer.Option(50, "--measurement-batches"),
    dataset_revision: str = typer.Option("v0.1.0-data", "--dataset-revision"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Measure inference throughput for a model across the benchmark's tasks."""
    try:
        model_cfg = load_yaml_with_schema(
            resolve_model_config(model), schemas_root() / "model_spec.json"
        )
    except FileNotFoundError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    task_list = list(tasks) if tasks else _enumerate_ranked_tasks(benchmark, language)
    if not task_list:
        typer.echo(_red(f"no ranked tasks found for {benchmark}/{language}"))
        raise typer.Exit(code=1)

    samples: list[tuple[str, ThroughputSample]] = []
    for task_name in task_list:
        try:
            task_cfg = load_yaml_with_schema(
                resolve_task_config(benchmark, task_name),
                schemas_root() / "task_spec.json",
            )
        except FileNotFoundError as exc:
            typer.echo(_red(str(exc)))
            raise typer.Exit(code=1) from exc

        try:
            repo_id = resolve_dataset_repo(task_cfg, language, prefer="private")
        except DatasetRepoError as exc:
            typer.echo(_red(str(exc)))
            raise typer.Exit(code=1) from exc
        token = resolve_hf_token()

        from balkanbench.cli import throughput as _self

        datasets = _self.load_dataset(
            repo_id,
            task_cfg["dataset"]["config"],
            revision=dataset_revision,
            token=token,
        )
        encoder = HFEncoder.build(model_cfg=model_cfg, task_cfg=task_cfg)

        eval_split_name = "validation" if "validation" in datasets else "test"

        # Build the predict_fn with task_type + num_choices already pinned so
        # the default_predict_fn picks the right input shape (2D for
        # classification, 3D for multiple_choice). Tests that monkeypatch
        # default_predict_fn still win because the helper resolves it at
        # call time.
        predict_fn = _task_scoped_predict_fn(
            task_type=task_cfg["task_type"],
            num_choices=int(task_cfg.get("num_choices", 2)),
        )

        sample = measure_task_throughput(
            model=encoder.model,
            tokenizer=encoder.tokenizer,
            task_cfg=task_cfg,
            dataset=datasets[eval_split_name],
            language=language,
            hardware=hardware,
            precision=precision,
            warmup_batches=warmup_batches,
            measurement_batches=measurement_batches,
            predict_fn=predict_fn,
        )
        write_task_throughput(
            sample=sample,
            out_dir=out,
            task=task_name,
            model=model_cfg["name"],
            model_id=model_cfg["hf_repo"],
            benchmark=benchmark,
            language=language,
        )
        samples.append((task_name, sample))
        typer.echo(
            _green(
                f"{task_name}: {sample.throughput_ex_per_sec:.1f} ex/s, "
                f"{sample.peak_vram_mib:.0f} MiB peak"
            )
        )

    write_model_throughput_aggregate(
        samples=samples,
        out_dir=out,
        model=model_cfg["name"],
        model_id=model_cfg["hf_repo"],
        benchmark=benchmark,
        language=language,
        hardware=hardware,
        precision=precision,
    )
    typer.echo(_green(f"Wrote aggregate to {out}"))
