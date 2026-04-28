"""Per-task and per-model throughput artifact writers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from balkanbench.throughput.measure import ThroughputSample

_TASK_SCHEMA = Path(__file__).resolve().parents[3] / "schemas" / "task_throughput.json"


def write_task_throughput(
    *,
    sample: ThroughputSample,
    out_dir: Path,
    task: str,
    model: str,
    model_id: str,
    benchmark: str,
    language: str,
    sponsor: str = "Recrewty",
    image_digest: str | None = None,
) -> Path:
    """Write a schema-valid per-task throughput JSON.

    Path: ``{out_dir}/{benchmark}-{language}/{model}/throughput/{task}.json``.
    """
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "language": language,
        "task": task,
        "model": model,
        "model_id": model_id,
        "hardware": sample.hardware,
        "precision": sample.precision,
        "batch_size": sample.batch_size,
        "max_seq_len": sample.max_seq_len,
        "warmup_batches": sample.warmup_batches,
        "measurement_batches": sample.measurement_batches,
        "throughput_ex_per_sec": sample.throughput_ex_per_sec,
        "throughput_tok_per_sec": sample.throughput_tok_per_sec,
        "peak_vram_mib": sample.peak_vram_mib,
        "torch_version": sample.torch_version,
        "driver_version": sample.driver_version,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sponsor": sponsor,
    }
    if image_digest or os.environ.get("BALKANBENCH_IMAGE_DIGEST"):
        payload["image_digest"] = image_digest or os.environ["BALKANBENCH_IMAGE_DIGEST"]

    _validate(payload)

    target_dir = Path(out_dir) / f"{benchmark}-{language}" / model / "throughput"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{task}.json"
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def write_model_throughput_aggregate(
    *,
    samples: list[tuple[str, ThroughputSample]],
    out_dir: Path,
    model: str,
    model_id: str,
    benchmark: str,
    language: str,
    hardware: str,
    precision: str,
) -> Path:
    """Aggregate per-task samples into a single ``{model}/throughput.json``.

    ``samples`` is a list of ``(task_name, ThroughputSample)`` pairs.
    Path: ``{out_dir}/{benchmark}-{language}/{model}/throughput.json``.
    """
    if not samples:
        raise ValueError("write_model_throughput_aggregate requires at least one sample")

    tasks_block: dict[str, dict[str, float]] = {}
    ex_per_sec_values: list[float] = []
    peak_vrams: list[float] = []
    for task, sample in samples:
        tasks_block[task] = {
            "ex_per_sec": sample.throughput_ex_per_sec,
            "peak_vram_mib": sample.peak_vram_mib,
        }
        ex_per_sec_values.append(sample.throughput_ex_per_sec)
        peak_vrams.append(sample.peak_vram_mib)

    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "language": language,
        "model": model,
        "model_id": model_id,
        "hardware": hardware,
        "precision": precision,
        "tasks": tasks_block,
        "mean_ex_per_sec": sum(ex_per_sec_values) / len(ex_per_sec_values),
        "max_peak_vram_mib": max(peak_vrams) if peak_vrams else 0.0,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sponsor": "Recrewty",
    }

    target_dir = Path(out_dir) / f"{benchmark}-{language}" / model
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "throughput.json"
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def _validate(payload: dict[str, Any]) -> None:
    schema = json.loads(_TASK_SCHEMA.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        ]
        raise ValueError("throughput artifact failed schema:\n" + "\n".join(messages))
