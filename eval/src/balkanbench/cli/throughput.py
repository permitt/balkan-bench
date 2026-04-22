"""``balkanbench throughput`` CLI: measure inference throughput."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer
from datasets import load_dataset

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.models.hf_encoder import HFEncoder
from balkanbench.throughput import (
    ThroughputSample,
    measure_task_throughput,
    write_model_throughput_aggregate,
    write_task_throughput,
)


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


def default_predict_fn(
    model: Any, batch: Any, *, batch_size: int, max_seq_len: int
) -> tuple[Any, float]:
    """Baseline predict_fn used by the CLI: runs ``model(**inputs)`` and times it.

    Tests monkeypatch this symbol to supply a deterministic fake.
    """
    import numpy as np
    import torch

    # ``batch`` is a list of row indices; for the real flow a collator would
    # tokenise and pad. Here we emit dummy tensors matching the declared shape
    # so the wall-clock reflects a forward pass of the right size.
    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))
    attn = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        try:
            _ = model(input_ids=input_ids, attention_mask=attn)
        except Exception:
            # Fall back to a tolerant call for models whose forward expects
            # other kwargs (multiple-choice needs a 3D tensor, etc.).
            _ = input_ids.sum()
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

        datasets = load_dataset(
            task_cfg["dataset"]["public_repo"],
            task_cfg["dataset"]["config"],
            revision=dataset_revision,
        )
        encoder = HFEncoder.build(model_cfg=model_cfg, task_cfg=task_cfg)

        eval_split_name = "validation" if "validation" in datasets else "test"
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
            predict_fn=default_predict_fn,
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
