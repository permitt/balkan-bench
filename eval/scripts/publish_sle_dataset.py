#!/usr/bin/env python3
"""Download, normalize, and publish the Serbian LLM Eval (SLE) dataset.

Source: ``gordicaleksa/serbian-llm-eval-v1`` (raw JSONL partials; its Python
loading script is not usable by modern ``datasets``, so we read the JSONL
files directly and normalize rows ourselves via
``balkanbench.data.sle_normalize.normalize_rows``).

Target: ``permitt/serbian-llm-eval`` (public), one config per task, with a
``test`` split for every task and an additional ``train`` split for
``nq_open`` and ``triviaqa``.

This script is independent of the SuperGLUE publish pipeline
(``balkanbench.data.publish``) - it does not import or reuse it.

Usage:
    HF_TOKEN=... uv run python scripts/publish_sle_dataset.py --dry-run
    HF_TOKEN=... uv run python scripts/publish_sle_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from balkanbench.data.sle_normalize import normalize_rows

SOURCE_REPO = "gordicaleksa/serbian-llm-eval-v1"
DEFAULT_TARGET_REPO = "permitt/serbian-llm-eval"

TASKS: list[str] = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "nq_open",
    "openbookqa",
    "piqa",
    "triviaqa",
    "winogrande",
]

# Only these two tasks have a train split upstream.
TRAIN_SPLIT_TASKS = {"nq_open", "triviaqa"}

# Hard-fail expected counts, per the task-4 brief. Any mismatch aborts the run.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "arc_challenge": {"test": 1172},
    "arc_easy": {"test": 2376},
    "boolq": {"test": 3270},
    "hellaswag": {"test": 10042},
    "openbookqa": {"test": 500},
    "piqa": {"test": 1838},
    "winogrande": {"test": 1267},
    "nq_open": {"test": 3610, "train": 87925},
    "triviaqa": {"test": 17944, "train": 138384},
}

# Upstream filenames follow a `{task}_{split}_partial_0_{N}_end.jsonl`
# convention (one file, `nq_open_test_..._end_end.jsonl`, has a duplicated
# trailing `_end` - handled by the optional group below).
FILENAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_(?P<split>test|train)_partial_0_(?P<n>\d+)_end(?:_end)?\.jsonl$"
)

CITATION_BLOCK = (
    "@article{serbian-llm-eval,\n"
    '  author    = "Gordić Aleksa",\n'
    '  title     = "Serbian LLM Eval",\n'
    '  year      = "2023"\n'
    "  howpublished = {\\url{https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1}},\n"
    "}"
)


class PublishError(RuntimeError):
    """Raised when the publish flow cannot proceed."""


def map_source_filenames(filenames: list[str]) -> dict[tuple[str, str], str]:
    """Map raw upstream JSONL filenames to ``(task, split) -> filename``.

    Files that don't match the naming convention or reference a task outside
    ``TASKS`` are ignored (e.g. ``README.md``, the loading script).
    """
    mapping: dict[tuple[str, str], str] = {}
    for name in filenames:
        m = FILENAME_RE.match(name)
        if m is None:
            continue
        task = m.group("task")
        split = m.group("split")
        if task not in TASKS:
            continue
        mapping[(task, split)] = name
    return mapping


def required_source_keys() -> list[tuple[str, str]]:
    """Every (task, split) pair we expect to find upstream."""
    keys: list[tuple[str, str]] = []
    for task in TASKS:
        keys.append((task, "test"))
        if task in TRAIN_SPLIT_TASKS:
            keys.append((task, "train"))
    return keys


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def download_and_normalize(
    *,
    task: str,
    split: str,
    filename: str,
    data_dir: Path,
) -> list[dict]:
    local_path = hf_hub_download(
        SOURCE_REPO, filename, repo_type="dataset", local_dir=str(data_dir)
    )
    raw_rows = load_jsonl(Path(local_path))
    return normalize_rows(task, split, raw_rows)


def assert_expected_counts(counts: dict[str, dict[str, int]]) -> None:
    problems = []
    for task, split_counts in EXPECTED_COUNTS.items():
        for split, expected in split_counts.items():
            actual = counts.get(task, {}).get(split)
            if actual != expected:
                problems.append(f"{task}/{split}: expected {expected}, got {actual}")
    if problems:
        raise PublishError(
            "row count mismatch vs. expected counts:\n  " + "\n  ".join(problems)
        )


def print_count_table(counts: dict[str, dict[str, int]]) -> None:
    print("\nPer-config row counts:")
    print(f"{'config':<15} {'split':<8} {'rows':>10}")
    for task in TASKS:
        for split in sorted(counts.get(task, {})):
            print(f"{task:<15} {split:<8} {counts[task][split]:>10}")


def render_readme_body(
    *,
    target_repo: str,
    upstream_revision: str,
    counts: dict[str, dict[str, int]],
) -> str:
    counts_rows = []
    for task in TASKS:
        for split in sorted(counts.get(task, {})):
            counts_rows.append(f"| `{task}` | `{split}` | {counts[task][split]} |")
    counts_table = "\n".join(counts_rows)

    return f"""\
# Serbian LLM Eval

This dataset is a republish of
[gordicaleksa/serbian-llm-eval-v1](https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1)
as plain Parquet configs, so it can be loaded with modern versions of the
`datasets` library (the upstream repo ships a Python loading script that
`datasets` can no longer execute). Row contents are otherwise unchanged; only
the `example_id` field is synthesized where the upstream data has no unique
identifier, and the `triviaqa` `answer` struct is flattened into
`answer_value` / `answer_aliases` columns.

## Attribution

All content originates from
[Aleksa Gordić](https://www.linkedin.com/in/aleksagordic)'s
**Serbian LLM Eval** project: English benchmark data machine-translated to
Serbian (Google Translate), refined via GPT-4, with manual review. See the
[upstream dataset card](https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1)
and the
[project report](https://wandb.ai/gordicaleksa/serbian_llm_eval/reports/First-Serbian-LLM-eval---Vmlldzo2MjgwMDA5)
for full methodology and credits.

- **Upstream repo:** [gordicaleksa/serbian-llm-eval-v1](https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1)
- **Upstream revision:** `{upstream_revision}`
- **License:** Apache 2.0 (matches upstream)

## Splits and counts

| Config | Split | Rows |
|--------|-------|------|
{counts_table}

Only `nq_open` and `triviaqa` have a `train` split; every other config has a
single `test` split.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{target_repo}", "arc_challenge", split="test")
```

## Citation

Please cite the upstream dataset:

```
{CITATION_BLOCK}
```

## License

Apache 2.0, matching the upstream `gordicaleksa/serbian-llm-eval-v1` dataset.
"""


def merge_readme(existing_readme: str, body: str, *, license_id: str = "apache-2.0") -> str:
    """Merge our custom body into the README ``push_to_hub`` generated.

    Keeps the YAML frontmatter (it carries the ``configs:`` / ``dataset_info``
    metadata ``datasets`` needs to find each config's parquet files) and
    ensures a ``license:`` field is present, but replaces everything below the
    frontmatter with our own documentation.
    """
    stripped = existing_readme.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            frontmatter = stripped[: end + len("\n---")]
            if "license:" not in frontmatter:
                frontmatter = frontmatter.replace("---\n", f"---\nlicense: {license_id}\n", 1)
            return f"{frontmatter}\n\n{body}"
    # No parseable frontmatter found (unexpected) - fall back to a minimal one.
    return f"---\nlicense: {license_id}\n---\n\n{body}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_TARGET_REPO,
        help=f"Target HF dataset repo id (default: {DEFAULT_TARGET_REPO})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download, normalize, and validate counts; skip push_to_hub and card upload.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory to download raw JSONL files into (default: a fresh temp dir).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_dir = (
        Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="sle_dataset_"))
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {SOURCE_REPO}...")
    source_files = list_repo_files(SOURCE_REPO, repo_type="dataset")
    filename_map = map_source_filenames(source_files)

    missing = [key for key in required_source_keys() if key not in filename_map]
    if missing:
        raise PublishError(f"could not find upstream files for: {missing}")

    upstream_info = HfApi().dataset_info(SOURCE_REPO)
    upstream_revision = upstream_info.sha
    print(f"Upstream revision: {upstream_revision}")

    from datasets import Dataset, DatasetDict

    prepared: dict[str, DatasetDict] = {}
    counts: dict[str, dict[str, int]] = {}

    for task in TASKS:
        splits = ["test"] + (["train"] if task in TRAIN_SPLIT_TASKS else [])
        split_datasets = {}
        for split in splits:
            filename = filename_map[(task, split)]
            print(f"Downloading + normalizing {task}/{split} ({filename})...")
            rows = download_and_normalize(
                task=task, split=split, filename=filename, data_dir=data_dir
            )
            split_datasets[split] = Dataset.from_list(rows)
            counts.setdefault(task, {})[split] = len(rows)
        prepared[task] = DatasetDict(split_datasets)

    assert_expected_counts(counts)
    print_count_table(counts)

    if args.dry_run:
        print("\n--dry-run: skipping push_to_hub and card upload.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PublishError("HF_TOKEN is not set; export it before running the real publish.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=False, exist_ok=True)

    for task, dataset_dict in prepared.items():
        print(f"Pushing config {task!r} to {args.repo}...")
        dataset_dict.push_to_hub(args.repo, config_name=task, token=token)

    print("Rendering + uploading dataset card...")
    existing_readme = hf_hub_download(args.repo, "README.md", repo_type="dataset", token=token)
    body = render_readme_body(
        target_repo=args.repo, upstream_revision=upstream_revision, counts=counts
    )
    merged_readme = merge_readme(Path(existing_readme).read_text(encoding="utf-8"), body)

    import io

    api.upload_file(
        path_or_fileobj=io.BytesIO(merged_readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )

    print(f"\nPublished {len(prepared)} configs to https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
