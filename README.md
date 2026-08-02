# BalkanBench

An open, reproducible benchmark and leaderboard for language models across
Serbian, Croatian, Montenegrin, and Bosnian (BCMS).

[balkanbench.com](https://balkanbench.com) - public leaderboard and launch page.

> Public release: **2026-04-28**. BalkanBench v1.0 launches a South Slavic
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

At launch, BalkanBench **v1.0** covers 3 released languages. The Serbian
SuperGLUE track is the official frozen track and ships with **6 ranked NLU
tasks** (BoolQ, CB, COPA, RTE, MultiRC, WSC) plus **2 diagnostics** (AX-b,
AX-g), totalling **67,313 items** across train, validation, and held-out test
splits. Croatian and Montenegrin release alongside as 5-task previews (no WSC
adaptation yet). The public leaderboard ships with **9 evaluated models**,
each run across **5 fixed seeds** on the held-out test split.

## Vision

BalkanBench is intended to become the central open-source benchmark hub for the
BCMS AI ecosystem: one place to publish datasets, compare models, reproduce
results, and collaborate on new evaluation tracks.

The roadmap extends beyond SuperGLUE. Serbian-LLM-Eval (SLE) has shipped
as a second track - see [below](#serbian-llm-eval-sle-track). Planned next
steps include:

- retrieval and embedding evaluation tracks
- Bosnian localization
- community-submitted benchmarks such as sentiment, NER, and domain-specific
  evaluation suites

## Languages (v1.0)

| Code | Language    | Status     |
|------|-------------|------------|
| sr   | Serbian     | official   |
| hr   | Croatian    | preview    |
| mne  | Montenegrin | preview    |
| bs   | Bosnian     | roadmap    |

## Ranked tasks (v1.0, SuperGLUE)

| Task    | sr  | hr  | mne |
|---------|:---:|:---:|:---:|
| BoolQ   | yes | yes | yes |
| CB      | yes | yes | yes |
| COPA    | yes | yes | yes |
| RTE     | yes | yes | yes |
| MultiRC | yes | yes | yes |
| WSC     | yes | -   | -   |

WSC is Serbian-only in v1.0 (no published HR/MNE adaptation yet), so
the Croatian and Montenegrin previews expose 5 ranked tasks each;
Serbian is the full 6-task SuperGLUE track. Diagnostics (AX-b, AX-g)
are Serbian-only and don't enter the ranked average.

## Serbian LLM Eval (SLE) track

BalkanBench ships a second Serbian track, adapted from
[Aleksa Gordić](https://www.linkedin.com/in/aleksagordic)'s
**Serbian LLM Eval** project (Apache 2.0), with permission and guidance
from Aleksa. It is **9 generative tasks** - arc_challenge, arc_easy, boolq,
hellaswag, openbookqa, piqa, winogrande, nq_open, triviaqa - translated and
adapted from English QA / commonsense-reasoning benchmarks. Upstream data:
[gordicaleksa/serbian-llm-eval-v1](https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1).
Our re-hosted parquet copy (full attribution, labels public):
[permitt/serbian-llm-eval](https://huggingface.co/datasets/permitt/serbian-llm-eval).
If you use this data, please cite the upstream dataset card.

SLE runs a **dual scoring protocol**, because open-weights and closed API
models can't be scored the same way:

- **Open-weights models** are scored via **loglikelihood** - a faithful
  port of EleutherAI's `lm-evaluation-harness` v0.3.0 as vendored in
  Gordić's fork, including its acc_norm character-length normalization,
  BoolQ/WinoGrande prompt construction, and context/continuation
  tokenization quirks.
- **Closed API models** (Claude, GPT, Gemini) don't expose the
  loglikelihood-over-continuation primitive that protocol needs, so they
  are scored via a **generative** protocol instead: multiple-choice tasks
  are reformulated as letter-choice prompts and the model's free-text
  answer is parsed.

These two protocols produce **two separate leaderboard tables that are
never comparable to each other**, even where a column shares a name (an
open-weights `acc_norm` and an API `acc` measure fundamentally different
things).

### Sequence-start tokens

Loglikelihood scoring is sensitive to whether a `<bos>`-style token opens
the sequence, and models disagree about what they expect. We therefore
score every model with **the prefix its own publisher specifies**, set
per model via `generation.prepend_bos`:

- Gemma 4 (all sizes) is trained with `<bos>` always present, so it gets
  one. Without it, every Gemma sits at chance on multiple choice: adding
  it moved Gemma-4-E4B by +8.8 (OpenBookQA) and +14.8 (ARC-Challenge).
- Mistral-derived models (Ministral, YugoGPT) declare
  `add_bos_token: true`, so they get one too. Ministral gained +13.0 and
  +10.8 on the same two tasks.
- Everything else is scored without one, which is what those models
  expect. This is not cosmetic: Granite's `bos_token` is an
  end-of-text marker, and prepending it *costs* -12.8 and -13.9.

Qwen, SmolLM3 and Slava define no BOS token at all and are unaffected
either way. The flag never changes which items are scored or how metrics
are computed - only the input format each model receives.

The open-weights board is live with **17 models**; the closed-API board
ships empty until those runs complete.

Official SLE runs pin the dataset to the `v1.0.0-sle-data` tag rather than
`main` - pass `--dataset-revision v1.0.0-sle-data` to `balkanbench eval`.

## Why it exists

BalkanBench started from a practical problem: we needed a reliable way to rank
local BCMS encoder models for real product use, instead of relying on scattered
claims or one-off internal tests.

That work began inside Recrewty's HR-tech efforts and grew into a broader
benchmarking initiative: if the English ecosystem benefits from shared
benchmarks and public leaderboards, the BCMS ecosystem should have the same.

## Hidden test labels

Each SuperGLUE language track declares a public HuggingFace dataset for
train/validation and public test inputs, plus a gated private sibling repo that
carries the hidden test labels used for official scoring. Public users can tune
and evaluate on labeled public train/validation data, generate public test
predictions with `balkanbench predict`, and submit those predictions for
trusted scoring. `balkanbench score` is the only path that needs the private
test labels.

The SLE track is the exception: its test labels were already public in the
upstream dataset, so the re-hosted copy keeps them public and `balkanbench
predict` is not supported for SLE tasks - `balkanbench eval` scores directly.

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
- Serbian LLM Eval dataset (re-host): <https://huggingface.co/datasets/permitt/serbian-llm-eval>
- Serbian LLM Eval dataset (upstream, Aleksa Gordić): <https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1>
- GitHub repository: <https://github.com/permitt/balkan-bench>

## License

MIT. See [LICENSE](LICENSE).

## Sponsor

Compute for official evaluation is sponsored by
**[Recrewty](https://recrewty.com)**.

## Contact

If you want to contribute a benchmark, model, language adaptation, or sponsor
future evaluation runs:

- Official email: <mailto:balkanbench@recrewty.com>

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
