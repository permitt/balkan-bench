# Security

## Supported versions

BalkanBench is on v0.1. Security fixes land on `main` and in the next
patch release. There is no long-term support branch for v0.1.

## Reporting a vulnerability

Email [perovicmitar@gmail.com](mailto:perovicmitar@gmail.com) with subject
line `BalkanBench security: <short topic>`. Please include:

- reproduction steps or proof-of-concept
- which commit or version is affected
- any suggested mitigation

We aim to acknowledge within 72 hours and publish a fix within 14 days
for high-severity issues. Please do not file public issues for
undisclosed vulnerabilities.

## Secret handling

Only one secret is handled by BalkanBench: **`HF_OFFICIAL_TOKEN`**. It
gates read access to the private test-labels Hugging Face repo
(`permitt/superglue-private`) and write access to the public dataset
repo during publishing.

Rules:

- Never bake the token into the Docker image.
- Never log the token (the CLI + GCP launchers redact it).
- Never commit the token to git.
- Store it in Secret Manager (GCP) or as a GitHub Actions secret.
- Rotate on any suspected compromise and at least every 90 days.

Details: [`docs/gcp/security.md`](docs/gcp/security.md).

## Integrity of official scoring

Official leaderboard rows come from `balkanbench score`, which refuses
to run without `HF_OFFICIAL_TOKEN` and fails loudly on any
example-id mismatch between the predictions file and the private labels.
There is no "score against public labels" fallback.

## Dependencies

Dependencies are pinned in `eval/pyproject.toml`. Dependabot is not wired
for v0.1; upgrades land via explicit PRs.

## Scope

In scope for reports: anything in `eval/src/balkanbench/`, `frontend/src/`,
schemas, CI workflows, GCP launcher scripts.

Out of scope: third-party services (HuggingFace, GCP, Vercel).
Report those to the upstream maintainers.
