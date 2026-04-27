# BalkanBench

An open, reproducible benchmark and leaderboard for language models across
Serbian, Croatian, Montenegrin, and Bosnian (BCMS).

[balkanbench.com](https://balkanbench.com) - public leaderboard and launch page.

> Public release: **2026-04-27**. BalkanBench v0.1 launches a South Slavic
> evaluation suite built around SuperGLUE, with Serbian as the official frozen
> track and Croatian plus Montenegrin released alongside it as open preview
> expansions.
>
> Compute for official evaluation is sponsored by [Recrewty](https://recrewty.com).

---

## What is this?

Over 20 million people speak Serbian, Montenegrin, Croatian, and Bosnian, yet
there has been no single public benchmark where local encoder models can be
compared on a shared evaluation suite.

BalkanBench is the first step toward that goal. It is two things living in one
repo:

1. **A benchmark contract**: a set of public datasets, hidden test labels, a
   scoring pipeline, and a frozen methodology that produces comparable,
   defensible model scores.
2. **An open-source framework** (`balkanbench`, this Python package) that runs
   the benchmark locally or on GCP, produces reproducible result artifacts, and
   exports a leaderboard JSON the frontend renders.

The repository is designed for contributions. Adding a new benchmark, task,
language, or model should be a schema-validated PR, not a core-code rewrite.

At launch, BalkanBench v0.1 covers 3 released languages, up to 7 NLU tasks plus
diagnostics, and more than 42k items across the released SuperGLUE datasets.
The public leaderboard ships with 9 evaluated models, each run across 5 fixed
seeds on the held-out test split.

## Vision

BalkanBench is intended to become the central open-source benchmark hub for the
BCMS AI ecosystem: one place to publish datasets, compare models, reproduce
results, and collaborate on new evaluation tracks.

The roadmap extends beyond SuperGLUE. Planned next steps include:

- Serbian-LLM-Eval, with permission and guidance from Aleksa Gordić
- retrieval and embedding evaluation tracks
- Bosnian localization
- community-submitted benchmarks such as sentiment, NER, and domain-specific
  evaluation suites

## Languages (v0.1)

| Code | Language    | Status     |
|------|-------------|------------|
| sr   | Serbian     | official   |
| hr   | Croatian    | preview    |
| mne  | Montenegrin | preview    |
| bs   | Bosnian     | roadmap    |

## Ranked tasks (v0.1, SuperGLUE)

| Task    | sr  | hr  | mne |
|---------|:---:|:---:|:---:|
| BoolQ   | yes | yes | yes |
| CB      | yes | yes | yes |
| COPA    | yes | yes | yes |
| RTE     | yes | yes | yes |
| MultiRC | yes | yes | yes |
| WSC     | yes | -   | -   |

WSC is Serbian-only in v0.1 (no published HR/MNE adaptation yet), so
the Croatian and Montenegrin previews expose 5 ranked tasks each;
Serbian is the full 6-task SuperGLUE track. Diagnostics (AX-b, AX-g)
are Serbian-only and don't enter the ranked average.

## Why it exists

BalkanBench started from a practical problem: we needed a reliable way to rank
local BCMS encoder models for real product use, instead of relying on scattered
claims or one-off internal tests.

That work began inside Recrewty's HR-tech efforts and grew into a broader
benchmarking initiative: if the English ecosystem benefits from shared
benchmarks and public leaderboards, the BCMS ecosystem should have the same.

## Hidden test labels

Each language track declares a public HuggingFace dataset for train/validation
and public test inputs, plus a gated private sibling repo that carries the
hidden test labels used for official scoring. Public users can tune and evaluate
on labeled public train/validation data, generate public test predictions with
`balkanbench predict`, and submit those predictions for trusted scoring.
`balkanbench score` is the only path that needs the private test labels.

## Quickstart

```bash
git clone https://github.com/permitt/balkan-bench
cd balkan-bench/eval

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

balkanbench --version
balkanbench list benchmarks
balkanbench list tasks
balkanbench list languages
balkanbench validate-env
```

The frontend is a separate, self-contained Vite app in `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

## Repository layout

```
balkan-bench/
├── frontend/            React landing + /leaderboard (Vercel rootDirectory)
├── eval/                Python package `balkanbench`
│   ├── src/balkanbench/ CLI, benchmarks, tasks, metrics, models, scoring, ...
│   ├── configs/         benchmark + task + model YAMLs
│   ├── schemas/         JSON Schemas that validate every config and artifact
│   ├── scripts/         dataset publisher, GCP launchers, aggregators
│   └── tests/           unit + integration + smoke
├── docs/                methodology, governance, leaderboard, GCP, specs
├── .github/             issue templates, workflows
└── README.md / CONTRIBUTING.md / LICENSE
```

## Contributing

We want this to be the benchmark the BCMS NLP community owns together. You do
not need core-code access to add a new benchmark, a new task inside an existing
benchmark, a new model, or to submit a result for an existing model.

**Four ways to contribute:**

| What | How |
|------|-----|
| Add a new **benchmark** (a new dataset + tasks, e.g. Croatian sentiment) | Open a `Propose Benchmark` issue, then a PR with `configs/benchmarks/<name>/`. Walkthrough in [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-benchmark). |
| Add a new **task** inside an existing benchmark | Open a `Propose Task` issue, then a PR with a new `tasks/<task>.yaml`. |
| Add a new **model** (leaderboard entry) | Open a `Propose Model` issue, then a PR with `configs/models/<model>.yaml` and an official result run. |
| **Submit a run** with predictions for an existing model + benchmark | Open a `Submission` issue with a `predictions.jsonl` package reference. |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide,
including a step-by-step walkthrough of adding a brand-new benchmark from
scratch.

## Resources

- Website and leaderboard: [balkanbench.com](https://balkanbench.com)
- Serbian SuperGLUE dataset: <https://huggingface.co/datasets/permitt/superglue-sr>
- Montenegrin SuperGLUE dataset: <https://huggingface.co/datasets/permitt/superglue-mne>
- Croatian SuperGLUE dataset: <https://huggingface.co/datasets/permitt/superglue-hr>
- GitHub repository: <https://github.com/permitt/balkan-bench>

## Full design and methodology

- [`docs/superpowers/specs/2026-04-22-balkanbench-v0.1-design.md`](docs/superpowers/specs/2026-04-22-balkanbench-v0.1-design.md)
  - the frozen v0.1 design
- [`docs/methodology/`](docs/methodology/) - benchmark contract, data provenance, versioning, task lifecycle, throughput
- [`docs/governance/`](docs/governance/) - submissions, contributions policy, anti-spam
- [`docs/leaderboard/format.md`](docs/leaderboard/format.md) - `benchmark_results.json` schema
- [`docs/gcp/`](docs/gcp/) - running official evaluation on GCP

(Some of the docs above are still being filled in during the v0.1 build.)

## License

MIT. See [LICENSE](LICENSE).

## Sponsor

Compute for official evaluation is sponsored by
**[Recrewty](https://recrewty.com)**.

## Contact

If you want to contribute a benchmark, model, language adaptation, or sponsor
future evaluation runs:

- Mitar Perović: <mailto:mitar@recrewty.com>
- LinkedIn: <https://linkedin.com/in/perovicmitar>

## Citation

If you use BalkanBench in research, please cite:

```bibtex
@misc{balkanbench2026,
  title   = {BalkanBench: An Open Evaluation Suite for BCMS Language Models},
  author  = {BalkanBench contributors},
  year    = {2026},
  url     = {https://balkanbench.com},
  note    = {Compute sponsored by Recrewty}
}
```
