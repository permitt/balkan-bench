# Contribution Governance

BalkanBench is open for community contributions. This document describes
the four contribution paths, the governance expectations for each, and
what maintainers look for during review. The hands-on how-to is in the
root [`CONTRIBUTING.md`](../../CONTRIBUTING.md); this file is the policy.

## Four contribution paths

| Path | Governance artifact |
|------|---------------------|
| New benchmark (new dataset or suite) | Propose-benchmark issue + PR with `configs/benchmarks/<name>/`. Authors become the benchmark's maintainers. |
| New task inside an existing benchmark | Propose-task issue + PR with `configs/benchmarks/<benchmark>/tasks/<task>.yaml`. |
| New model | Propose-model issue + PR with `configs/models/<tier>/<model>.yaml`. |
| Run for an existing (model, benchmark) | Submission issue. See [`submissions.md`](submissions.md). |

## Review criteria

Every proposal is reviewed against the same criteria:

1. **Identity**: a public GitHub or Hugging Face handle is present and
   reachable.
2. **License**: datasets ship with an SPDX-identified open license
   (typically CC-BY-4.0); models declare their license in the YAML.
3. **Schema conformance**: `balkanbench validate-config` passes in CI.
   The JSON Schemas under `eval/schemas/` are the source of truth.
4. **Reproducibility**: the proposed tooling can produce the same
   numbers twice with the same seed.
5. **Scope fit**: the proposal is on-topic for BCMS LLM evaluation; it
   does not silently broaden the benchmark's definition.
6. **Documentation**: the proposal updates the right docs in the same
   PR (the methodology file for a new task, the dataset card for a new
   benchmark, the model card for a new model).

## Maintainer duties

When a maintainer accepts a proposal they take on:

- answering questions on the tracking issue within the SLA below,
- keeping the corresponding YAMLs schema-valid as schemas evolve,
- running the reproducibility gate on the added work at least once per
  minor release cycle.

Benchmark-level maintainers are listed in each benchmark's
`benchmark.yaml` `maintainers` block and are expected to shepherd
tasks, models, and submissions within that benchmark. Benchmark-level
maintainers are not core maintainers unless explicitly designated.

## Conflict of interest

Maintainers may review + approve contributions from their own
affiliations, but must recuse from decisions where a commercial or
personal incentive could reasonably be perceived as biasing the
outcome. Recusal is logged in the tracking issue.

## Review SLAs

- Issue acknowledgement: within 72 hours.
- Triage decision (accept / reject / needs-more-info): within 7 days
  for benchmark / task / model proposals; within 7 days for
  submissions (see [`submissions.md`](submissions.md)).
- PR review: within 14 days of the PR being opened by the assigned
  reviewer.

## Decisions outside the schemas

Decisions that change the benchmark contract (adding a ranked task,
changing a metric, bumping major) are tracked in the relevant doc's
decision log + release notes, not just a PR description.

## Code of conduct

Short version: courteous, respectful behaviour in issues, PRs, reviews,
and any space that uses the BalkanBench name. Assume good faith,
disagree on content not on people, flag anything that crosses a line to
a maintainer.

A full code of conduct document lands alongside v0.1.0. Until it is in
the repo, reports go to
[perovicmitar@gmail.com](mailto:perovicmitar@gmail.com).
