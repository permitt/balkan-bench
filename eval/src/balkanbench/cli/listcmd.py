"""``balkanbench list ...`` discovery commands."""

from __future__ import annotations

import os
from pathlib import Path

import typer
import yaml

list_app = typer.Typer(
    name="list",
    help="Discover configured benchmarks, tasks, models, languages.",
    no_args_is_help=True,
    add_completion=False,
)


def _configs_root() -> Path:
    override = os.environ.get("BALKANBENCH_CONFIGS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs"


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*.yaml"))


def _collect_languages() -> dict[str, set[str]]:
    """Walk task YAMLs and group language codes by status (available / roadmap).

    Returns a dict with keys ``"available"`` and ``"roadmap"`` mapping to sets of
    language codes discovered across every benchmark's tasks.
    """
    groups: dict[str, set[str]] = {"available": set(), "roadmap": set()}
    benchmarks_dir = _configs_root() / "benchmarks"
    if not benchmarks_dir.is_dir():
        return groups
    for bench_dir in benchmarks_dir.iterdir():
        if not bench_dir.is_dir():
            continue
        for task_yaml in _yaml_files(bench_dir / "tasks"):
            try:
                doc = yaml.safe_load(task_yaml.read_text())
            except yaml.YAMLError:
                continue
            if not isinstance(doc, dict):
                continue
            langs = doc.get("languages") or {}
            for key in ("available", "roadmap"):
                for lang in langs.get(key) or []:
                    groups[key].add(str(lang))
    return groups


@list_app.command("benchmarks")
def list_benchmarks() -> None:
    """List known benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no benchmarks configured yet")
        return
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not names:
        typer.echo("no benchmarks configured yet")
        return
    for name in names:
        typer.echo(name)


@list_app.command("tasks")
def list_tasks() -> None:
    """List tasks across all benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no tasks configured yet")
        return
    task_ids: list[str] = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_yaml in _yaml_files(bench_dir / "tasks"):
            task_ids.append(f"{bench_dir.name}.{task_yaml.stem}")
    if not task_ids:
        typer.echo("no tasks configured yet")
        return
    for tid in task_ids:
        typer.echo(tid)


@list_app.command("models")
def list_models() -> None:
    """List model configs."""
    root = _configs_root() / "models"
    if not root.is_dir():
        typer.echo("no models configured yet")
        return
    files = _yaml_files(root)
    if not files:
        typer.echo("no models configured yet")
        return
    for f in files:
        tier = f.parent.name if f.parent.name in {"official", "experimental"} else "-"
        typer.echo(f"{f.stem}\t{tier}")


@list_app.command("languages")
def list_languages() -> None:
    """List languages discovered across every configured task."""
    groups = _collect_languages()
    if not groups["available"] and not groups["roadmap"]:
        typer.echo("no languages discovered (add task YAMLs under configs/benchmarks/*/tasks/)")
        return
    for lang in sorted(groups["available"]):
        typer.echo(f"{lang}\tavailable")
    for lang in sorted(groups["roadmap"] - groups["available"]):
        typer.echo(f"{lang}\troadmap")
