"""Assemble ``benchmark_results.json`` from on-disk official result artifacts.

The input layout (produced by ``balkanbench eval`` + ``balkanbench score``):

    {results_root}/{model}/{task}/result.json

Each file is a ``schemas/result_artifact.json``-valid artifact. The output
conforms to ``schemas/leaderboard_export.json``. Ranking rules:

- Row is ``complete=true`` only when every ranked task has a ``rankable=true``
  artifact with a primary metric.
- Only complete rows receive an integer rank (sorted by row ``avg`` descending,
  ties broken by model name).
- Partial rows (``rankable=false`` or missing tasks) keep ``rank=null`` and
  carry a ``partial_flag`` like ``"(5/6) partial"``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "leaderboard_export.json"


class ExportError(ValueError):
    """Raised when the leaderboard cannot be assembled."""


def assemble_leaderboard(
    *,
    benchmark: str,
    language: str,
    results_root: Path,
    ranked_tasks: list[str],
    task_primary_metrics: dict[str, str],
    benchmark_version: str,
    seeds: int = 5,
    sponsor: str = "Recrewty",
    throughput_meta: dict[str, Any] | None = None,
    access: str | None = None,
) -> dict[str, Any]:
    """Build the leaderboard payload; validates against the JSON Schema.

    ``access`` (``"open_weights"`` | ``"api"`` | ``None``) splits a shared
    results tree into the two SLE boards. When set, only artifacts whose
    ``run_config.access`` matches are included (artifacts without a
    ``run_config`` count as ``open_weights`` for backward compat); the export
    is stamped with ``access`` and, when the included artifacts carry it,
    ``protocol`` (all included artifacts must agree - a mismatch is a data
    error). ``seeds`` is dropped from the export when the included artifacts
    carry none (generative runs are single-pass); mixed presence is a data
    error. When ``access`` is ``None`` behavior is unchanged from before this
    parameter existed (no filtering, no ``protocol``/``access`` stamped,
    ``seeds`` always present) so existing exports stay byte-identical.
    """
    if not results_root.is_dir():
        raise ExportError(f"results_root does not exist: {results_root}")

    rows: list[dict[str, Any]] = []
    included_protocols: set[str] = set()
    included_has_seeds: set[bool] = set()
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        row, protocols, has_seeds_flags = _build_row(
            model_dir=model_dir,
            ranked_tasks=ranked_tasks,
            task_primary_metrics=task_primary_metrics,
            access=access,
        )
        included_protocols.update(protocols)
        included_has_seeds.update(has_seeds_flags)
        if row:
            rows.append(row)

    _assign_ranks(rows)

    export: dict[str, Any] = {
        "benchmark": benchmark,
        "language": language,
        "benchmark_version": benchmark_version,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sponsor": sponsor,
        "seeds": seeds,
        "ranked_tasks": list(ranked_tasks),
        "task_primary_metrics": dict(task_primary_metrics),
        "rows": rows,
    }
    if throughput_meta is not None:
        export["throughput"] = throughput_meta

    if access is not None:
        export["access"] = access
        if len(included_protocols) > 1:
            raise ExportError(
                "leaderboard export: included artifacts disagree on run_config.protocol: "
                f"{sorted(included_protocols)}"
            )
        if included_protocols:
            export["protocol"] = next(iter(included_protocols))
        if len(included_has_seeds) > 1:
            raise ExportError(
                "leaderboard export: included artifacts disagree on presence of 'seeds' "
                "(mix of seeded and single-pass generative runs)"
            )
        if included_has_seeds == {False}:
            del export["seeds"]

    _validate(export)
    return export


def write_leaderboard_export(
    *,
    benchmark: str,
    language: str,
    results_root: Path,
    ranked_tasks: list[str],
    task_primary_metrics: dict[str, str],
    benchmark_version: str,
    out_path: Path,
    seeds: int = 5,
    sponsor: str = "Recrewty",
    throughput_meta: dict[str, Any] | None = None,
    access: str | None = None,
) -> Path:
    """Assemble and write the export to ``out_path``; returns the path."""
    export = assemble_leaderboard(
        benchmark=benchmark,
        language=language,
        results_root=results_root,
        ranked_tasks=ranked_tasks,
        task_primary_metrics=task_primary_metrics,
        benchmark_version=benchmark_version,
        seeds=seeds,
        sponsor=sponsor,
        throughput_meta=throughput_meta,
        access=access,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export, indent=2))
    return out_path


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _build_row(
    *,
    model_dir: Path,
    ranked_tasks: list[str],
    task_primary_metrics: dict[str, str],
    access: str | None,
) -> tuple[dict[str, Any] | None, set[str], set[bool]]:
    """Build one leaderboard row from a model's per-task result artifacts.

    Returns ``(row, protocols, has_seeds_flags)``. ``row`` is ``None`` when no
    matching artifact is found for any ranked task (the model is skipped
    entirely). ``protocols``/``has_seeds_flags`` summarize the artifacts that
    were actually included (post ``access`` filtering) so the caller can
    stamp/validate the export-level ``protocol`` and ``seeds`` fields; they
    are collected unconditionally but only enforced by the caller when
    ``access`` is not ``None``.
    """
    model_slug = model_dir.name
    results: dict[str, dict[str, float] | None] = {t: None for t in ranked_tasks}
    model_id: str | None = None
    model_revision: str | None = None
    params: int | None = None
    display_name: str | None = None
    any_non_rankable = False
    tasks_present = 0
    protocols: set[str] = set()
    has_seeds_flags: set[bool] = set()

    for task in ranked_tasks:
        artifact_path = model_dir / task / "result.json"
        if not artifact_path.is_file():
            continue
        artifact = json.loads(artifact_path.read_text())
        run_config = artifact.get("run_config") or {}
        artifact_access = run_config.get("access", "open_weights")
        if access is not None and artifact_access != access:
            # Not part of this board; treat as if the artifact didn't exist.
            continue
        if "protocol" in run_config:
            protocols.add(run_config["protocol"])
        has_seeds_flags.add("seeds" in artifact)
        if not artifact.get("rankable", False):
            any_non_rankable = True
        metric_name = task_primary_metrics[task]
        mean = float(artifact["aggregate"]["mean"][metric_name])
        stdev = float(artifact["aggregate"]["stdev"].get(metric_name, 0.0))
        results[task] = {"mean": mean, "stdev": stdev}
        tasks_present += 1
        model_id = artifact["model_id"]
        model_revision = artifact.get("model_revision")
        params = artifact.get("params") or params
        # Prefer a display name on the artifact (so a pretty row label like
        # "BERTić" survives round-trips through the slug-based directory layout).
        display_name = artifact.get("model_display_name") or display_name

    if tasks_present == 0:
        # No result file found for any ranked task; skip this model entirely.
        return None, protocols, has_seeds_flags

    complete = (tasks_present == len(ranked_tasks)) and not any_non_rankable
    present_scores = [r["mean"] for r in results.values() if r is not None]
    avg = sum(present_scores) / len(present_scores) if present_scores else 0.0

    row: dict[str, Any] = {
        "rank": None,  # assigned by _assign_ranks
        "model": display_name or model_slug,
        "model_id": model_id or f"unknown/{model_slug}",
        "params": int(params or 0),
        "params_display": _fmt_params(params or 0),
        "results": results,
        "avg": round(avg, 4),
        "complete": complete,
        "tasks_completed": tasks_present,
        "tasks_total": len(ranked_tasks),
    }
    if access == "api":
        # API rows carry no open-weight parameter count.
        row["params"] = None
        row["params_display"] = "API"
    if model_revision:
        row["model_revision"] = model_revision
    if not complete:
        row["partial_flag"] = f"({tasks_present}/{len(ranked_tasks)}) partial"
    return row, protocols, has_seeds_flags


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    """Assign integer ranks to complete rows, leaving partials with rank=null."""
    complete = [r for r in rows if r["complete"]]
    complete.sort(key=lambda r: (-r["avg"], r["model"]))
    for i, row in enumerate(complete, start=1):
        row["rank"] = i


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _validate(export: dict[str, Any]) -> None:
    schema = json.loads(_SCHEMA_PATH.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(export), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        ]
        raise ExportError("leaderboard export failed schema:\n" + "\n".join(messages))
