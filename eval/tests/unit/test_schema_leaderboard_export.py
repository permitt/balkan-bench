"""Validate benchmark_results.json payloads."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "leaderboards"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "leaderboard_export.json").read_text())


def test_valid_export_passes() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(export)


def test_partial_row_allowed() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    export["rows"].append(
        {
            "rank": None,
            "model": "ModernBERTić small",
            "model_id": "permitt/modernbertic-small",
            "params": 149000000,
            "params_display": "149M",
            "results": {
                "cb": {"mean": 76.96, "stdev": 3.19},
                "copa": {"mean": 65.76, "stdev": 2.42},
                "rte": {"mean": 65.82, "stdev": 1.14},
                "wsc": {"mean": 64.11, "stdev": 1.11},
                "boolq": {"mean": 76.02, "stdev": 0.63},
                "multirc": None,
            },
            "avg": 69.73,
            "complete": False,
            "tasks_completed": 5,
            "tasks_total": 6,
            "partial_flag": "(5/6)",
            "throughput": {"ex_per_sec": 312.1, "peak_vram_mib": 2410},
        }
    )
    Draft202012Validator(_load_schema()).validate(export)


def test_missing_sponsor_fails() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    del export["sponsor"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(export))
    assert errors
