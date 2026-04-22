# Task Lifecycle

Every BalkanBench task is in one of four states. State transitions
require explicit governance; tasks do not drift between states silently.

## States

| Status         | On the leaderboard? | In the main score? | Allowed changes |
|----------------|---------------------|---------------------|------------------|
| `ranked`       | yes                 | yes (contributes)   | patch / minor    |
| `diagnostic`   | yes (dedicated column) | no               | patch / minor    |
| `experimental` | no                  | no                  | any              |
| `archived`     | no (history only)   | no                  | none (read-only) |

`status` is set in the task YAML under `status:` and enforced by the
JSON Schema. See
[`eval/schemas/task_spec.json`](../../eval/schemas/task_spec.json).

## Transition rules

### `experimental -> ranked`

Promotion from experimental to ranked is a **minor benchmark bump**.
Prerequisites:

- At least 3 published models have been run through the task via
  `balkanbench eval` (not just the task author's own model).
- No data-quality issues filed against the task in the previous minor
  release cycle.
- A maintainer has approved the promotion in the tracking issue.

The benchmark minor version increments on the release that flips the
status.

### `ranked -> archived`

Deprecating a ranked task is a **major benchmark bump**: it changes
official numbers on every existing ranked row because the main score
drops a column.

Prerequisites:

- A maintainer-approved deprecation notice filed at least 30 days
  before the release that flips the status.
- A replacement task (possibly a bug-fixed variant) is in `experimental`
  and on a promotion path, or the main-score formula is explicitly
  rebalanced in the same major release.

Historical ranked artifacts stay in the repo under
`eval/results/official/.../{model}/{task}/result.json` with a top-level
`archived-task` flag on the leaderboard. They never re-enter the main
score.

### `ranked / experimental -> diagnostic`

Minor bump if the task's contribution was small; major bump if it was
meaningful (reviewer judgment, documented in the release notes).

### anything -> `experimental`

Any task that fails the reproducibility gate in two consecutive minor
releases drops to `experimental` automatically. It re-enters `ranked`
only after passing the promotion rules above.

## v0.1 state

- `ranked`: BoolQ, CB, COPA, RTE, MultiRC, WSC.
- `diagnostic`: AX-b, AX-g.
- `experimental`: none in v0.1.
- `archived`: none in v0.1.

## ReCoRD

ReCoRD is **not in v0.1**. The previous-generation pipeline pinned every
model to EM ~0.17 regardless of architecture, which almost certainly
reflects a metric / grouping bug rather than real model behaviour.
ReCoRD will land in `experimental` once the metric is re-derived and
sanity-checked; it is a candidate for `ranked` in a later release after
the promotion rules above.

## COPA variants

Legacy `copa_fixed` and `copa_connective` variants from the v6/v10
codebase are **not** re-exposed as public task IDs. Only the canonical
`copa` is active. Historical variants may be described in methodology
notes but must not exist as active public configs or leaderboard
columns. See the v0.1 design spec decision log for the rationale.

## Governance

Every state transition happens via a GitHub issue of type
`proposal-task` or a dedicated `task-lifecycle` issue, reviewed by a
maintainer listed in the benchmark manifest's `maintainers` block. The
authoritative deprecation record lives in the release notes of the
release that flipped the status.
