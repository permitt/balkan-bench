# Submissions Governance

Official leaderboard rows require accountable submitters. This document
spells out what "official" means and what keeps the leaderboard spam-free
and defensible.

## Submitter identity

Every official submission must identify a submitter via a public handle:

- GitHub username, or
- Hugging Face username.

The handle must be reachable (profile exists, is not deleted, and ideally
has prior activity). Anonymous or throwaway-account submissions are
rejected. Identity is captured in the submission metadata JSON
([`eval/schemas/submission_metadata.json`](../../eval/schemas/submission_metadata.json)).

Rationale: this is the minimum bar for holding a result accountable when
someone else tries to reproduce it. It also gives maintainers a person
to email when review questions come up.

## Required fields

A submission metadata JSON must include:

- `submission_id` (UUIDv4 the tooling generates).
- `submitter.name` (how the submitter should be credited).
- `submitter.identity.provider` (`github` or `huggingface`).
- `submitter.identity.handle`.
- `model.name`, `model.hf_repo`, `model.license`.
- `benchmark.name`, `benchmark.version`.
- `predictions_package.path`, `predictions_package.sha256`
  (`sha256:<64hex>`).

Optional: `submitter.email`, `submitter.affiliation`, `model.hf_revision`,
`model.params`, `notes`.

## Submission flow

1. **Predict.** Run `balkanbench predict` against the public test split.
2. **Package.** `balkanbench submit results/local/... --out submission.json`
   produces a submission JSON + a predictions tarball.
3. **Upload.** Host the tarball on a permanent URL (Hugging Face,
   GitHub Release asset, or GCS).
4. **Open issue.** File a `Submission` issue with the submission JSON
   attached.
5. **Maintainer triage.** Identity, license, repo reachability, and
   reproducibility are checked.
6. **Score.** A maintainer runs `balkanbench score` with
   `HF_OFFICIAL_TOKEN` in the official environment.
7. **Commit.** The scored `result.json` is committed under
   `eval/results/official/{benchmark}-{language}/{model}/`. The
   leaderboard export is regenerated.

## Anti-spam rules

- Max 1 submission per model per week without a substantive reason
  (e.g. a new revision of the checkpoint).
- Submissions with fabricated or inaccessible model repos are rejected
  immediately; a second offence results in a cooldown.
- Predictions packages that fail the sha256 check are rejected.
- Predictions containing example IDs not present in the private labels,
  or missing ids present in the private labels, are rejected. Exact
  alignment is enforced by `balkanbench score`.

## Below-chance submissions

`balkanbench score` refuses to emit a diagnostic result that scores more
than 3 standard deviations below chance (see
`docs/methodology/benchmark_contract.md`). For ranked tasks, submissions
that score below chance are accepted but flagged on the leaderboard;
they often signal a label-mapping or tokenisation bug worth documenting.

## Review SLAs

- Issue acknowledgement: within 72 hours.
- Triage (accept / reject / needs-more-info): within 7 days.
- Scoring + commit for accepted submissions: within 14 days, budget
  permitting (GCP compute sponsored by Recrewty).

## Appeals

If a submission is rejected and the submitter disagrees, they may
re-file with the original issue number and additional evidence.
Maintainer decisions are final for the release they apply to; a
rejected submission from one release can be resubmitted unchanged in a
later release if policy changes.

## Retractions

A maintainer may retract an official row when a bug, license violation,
or fraudulent submission is discovered. Retractions are announced in
the release notes of the next release. The retracted artifact moves
to `eval/results/retracted/` with a link to the retraction issue.

## Contact

Governance questions: file a GitHub issue with the `governance` label
or email [perovicmitar@gmail.com](mailto:perovicmitar@gmail.com).
