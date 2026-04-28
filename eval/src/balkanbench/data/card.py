"""Render a Hugging Face dataset card (README.md) from a manifest."""

from __future__ import annotations

from typing import Any

_TEMPLATE = """---
license: {license}
language:
{language_yaml}
tags:
  - balkanbench
  - {benchmark}
  - bcms
  - evaluation
pretty_name: "BalkanBench {benchmark_upper} ({language_name})"
---

# BalkanBench {benchmark_upper} ({language_name})

**Revision:** `{revision}`
**License:** {license}
**Compute sponsored by [Recrewty](https://recrewty.com).**

This dataset is part of [BalkanBench](https://balkanbench.com), an open
benchmark for evaluating language models on BCMS languages (Serbian, Croatian,
Montenegrin, Bosnian). It is the {language_name} adaptation of the
**{benchmark_upper}** suite.

## Hidden test labels

{hidden_label_block}

## Splits and counts

| Config | Split | Rows | Has labels |
|--------|-------|------|------------|
{splits_table}

## Fields

{fields_block}

## Reproduction

```bash
pip install 'balkanbench[dev]'

# Generate predictions on the public test split
balkanbench predict \\
  --benchmark {benchmark} \\
  --language {language} \\
  --model <your-model-config>

# Official scoring (requires HF_OFFICIAL_TOKEN access to the private labels repo)
balkanbench score \\
  --benchmark {benchmark} \\
  --language {language} \\
  --predictions predictions.jsonl
```

## Source

- Public repo: [{public_repo}](https://huggingface.co/datasets/{public_repo})
{private_line}
- Full design and methodology: [balkanbench.com](https://balkanbench.com)

## Citation

```bibtex
@misc{{balkanbench2026,
  title   = {{BalkanBench: An Open Evaluation Suite for BCMS Language Models}},
  author  = {{BalkanBench contributors}},
  year    = {{2026}},
  url     = {{https://balkanbench.com}},
  note    = {{Compute sponsored by Recrewty}}
}}
```
"""

_LANG_NAMES: dict[str, str] = {
    "sr": "Serbian",
    "hr": "Croatian",
    "cnr": "Montenegrin",
    "bs": "Bosnian",
}

_HIDDEN_LABEL_TEXT = (
    "The `test` split of every config is published **without labels**. "
    "Public users can generate predictions with `balkanbench predict`, "
    "but test labels are hidden: `balkanbench score` requires access to a "
    "private label store and is intended to run only in the official scoring "
    "environment. This preserves leaderboard integrity."
)

_VISIBLE_LABEL_TEXT = (
    "All splits in this dataset ship with labels. See the config-specific "
    "fields table below for the label column name."
)


def render_dataset_card(manifest: dict[str, Any], *, sponsor: str = "Recrewty") -> str:
    """Render a dataset card (HF ``README.md``) from a schema-valid manifest.

    ``sponsor`` is retained as a parameter for forward compat; the template
    bakes in Recrewty because every v0.1 dataset is Recrewty-sponsored.
    """
    _ = sponsor  # reserved for future swap-outs
    language = manifest["language"]
    language_name = _LANG_NAMES.get(language, language)
    benchmark = manifest["benchmark"]

    splits_rows: list[str] = []
    fields_blocks: list[str] = []
    for config_name, config in manifest["configs"].items():
        for split_name, split_info in config["splits"].items():
            splits_rows.append(
                f"| `{config_name}` | `{split_name}` | "
                f"{split_info['num_rows']} | "
                f"{'yes' if split_info['has_labels'] else 'no'} |"
            )
        fields_blocks.append(
            f"### `{config_name}`\n\n" + ", ".join(f"`{f}`" for f in config["fields"])
        )

    private_line = (
        f"- Private labels repo (gated): `{manifest['private_repo']}`"
        if manifest.get("private_repo")
        else ""
    )
    hidden_label_block = (
        _HIDDEN_LABEL_TEXT if manifest["hidden_test_labels"] else _VISIBLE_LABEL_TEXT
    )

    return _TEMPLATE.format(
        license=manifest["license"],
        language_yaml=f"  - {language}",
        benchmark=benchmark,
        benchmark_upper=benchmark.upper(),
        language=language,
        language_name=language_name,
        revision=manifest["dataset_revision"],
        hidden_label_block=hidden_label_block,
        splits_table="\n".join(splits_rows),
        fields_block="\n\n".join(fields_blocks),
        public_repo=manifest["public_repo"],
        private_line=private_line,
    )
