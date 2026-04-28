# balkanbench (Python package)

The evaluation framework for [BalkanBench](https://balkanbench.com).

## Install (development)

```bash
cd eval
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick check

```bash
balkanbench --version
balkanbench --help
```

## Layout

```
src/balkanbench/   # package source
configs/           # benchmark, task, model configs
schemas/           # JSON Schemas for configs + artifacts
scripts/           # publish + aggregate + GCP launchers
tests/             # unit + integration + smoke
results/           # official results (committed); local/submissions gitignored
```

