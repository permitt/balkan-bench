# BalkanBench

An open, reproducible evaluation suite for large language models across
Serbian, Croatian, Montenegrin, and Bosnian (BCMS).

[balkanbench.com](https://balkanbench.com) - public leaderboard and launch page.

> v0.1 target release: **2026-04-27**. Scope: Serbian SuperGLUE is the
> official frozen track. Croatian and Montenegrin SuperGLUE are published as
> preview expansions, and Bosnian plus Serbian-LLM-Eval (generative, Aleksa
> Gordić) land in v0.2.
>
> Compute for official evaluation is sponsored by [Recrewty](https://recrewty.com).

---

## What is this?

BalkanBench is two things living in one repo:

1. **A benchmark contract**: a set of public datasets, hidden test labels, a
   scoring pipeline, and a frozen methodology that produces comparable,
   defensible model scores.
2. **An open-source framework** (`balkanbench`, this Python package) that runs
   the benchmark locally or on GCP, produces reproducible result artifacts, and
   exports a leaderboard JSON the frontend renders.

The repository is designed for contributions. Adding a new benchmark, task,
language, or model is a schema-validated PR, not a core-code rewrite.

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
