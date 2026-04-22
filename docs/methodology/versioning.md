# Versioning

BalkanBench uses semantic versioning with explicit rules for what forces
a major, minor, or patch bump. The benchmark version and the dataset
version are tracked separately.

## Benchmark version

The version recorded in every `result.json` artifact's `benchmark_version`
field and surfaced on the leaderboard.

| Bump level | Triggers |
|------------|----------|
| **major**  | Change to the ranked task list, to an official metric definition, to the main-score aggregation formula, or to the hidden/public scoring contract. Anything that can change official numbers on existing rows. |
| **minor**  | Addition of a new language, a new ranked task (with no change to existing tasks), a new diagnostic, or a non-breaking schema extension. |
| **patch**  | Bug fixes that do not change official leaderboard numbers, documentation improvements, CI changes, provenance clarifications. |

v0.1.0 is the initial public release. A future breaking change (e.g.
adopting a different COPA formulation, or changing `f1_a` to
`exact_match` as the MultiRC primary metric) would bump the benchmark
to `v1.0.0`; a purely additive change like launching Croatian data
bumps to `v0.2.0`.

## Dataset version

The Hugging Face dataset (`permitt/superglue-serbian`) uses its own
semver tag recorded in every artifact's `dataset_revision` field. Format:
`vX.Y.Z-data`.

| Bump level | Triggers |
|------------|----------|
| **major**  | Any change to test labels or test-split composition. |
| **minor**  | Additional examples appended to train or validation; new task config added. |
| **patch**  | Non-label metadata fixes (typos, field-name corrections, data card edits). |

Revisions are immutable once tagged. Relabelling test data requires a
major dataset version.

## Schemas

JSON Schemas are versioned in-place with the package. A breaking change
to a schema is a major benchmark bump because it can invalidate
previously-valid artifacts.

## Task-type enum

New `task_type` values added to the schema are a minor bump. Renaming
or removing an existing `task_type` is a major bump.

## Compatibility

Every artifact carries `benchmark_version`, `dataset_revision`, and
`code_revision` so a reader can reconstruct what was true at the time
the row was produced. The leaderboard export refuses to mix rows from
different benchmark majors.

## Release cadence

- v0.x minor releases are cut as features land and CI stays green.
- Major releases are announced at least 30 days ahead with a migration
  note covering the breaking change.
