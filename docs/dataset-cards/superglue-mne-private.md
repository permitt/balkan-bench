---
license: cc-by-4.0
language:
  - me      # sr, hr, or me
multilinguality: monolingual
pretty_name: BalkanBench SuperGLUE - Montenegrin (private test labels)
size_categories:
  - 10K<n<100K
source_datasets:
  - extended|super_glue
task_categories:
  - text-classification
  - multiple-choice
  - question-answering
extra_gated_prompt: |
  This repository carries the **hidden test labels** for the BalkanBench
  SuperGLUE-Montenegrin benchmark. To preserve leaderboard integrity, access is
  granted only to the official BalkanBench scoring environment.

  If you are a model author, you do **not** need access to this repo to
  submit a result. Use the public `permitt/superglue-mne` repo to
  generate test predictions, then submit them through the BalkanBench
  submission flow at https://balkanbench.com.
extra_gated_fields:
  Reason for access: text
  Affiliation: text
tags:
  - balkanbench
  - bcms
  - montenegrin       # serbian / croatian / montenegrin
  - superglue
  - benchmark
  - hidden-test-labels
---

# BalkanBench SuperGLUE - Montenegrin (private test labels)

> **Gated companion to** [`permitt/superglue-mne`](https://huggingface.co/datasets/permitt/superglue-mne).
> If you are a model author and just want to evaluate, use the **public**
> repo - you do not need access to this one. Live leaderboard:
> <https://balkanbench.com/leaderboard>. Background:
> [Release of BalkanBench - the vision behind it (Medium)](https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411).

This is the **gated sibling** of the public `permitt/superglue-mne`
repo. It carries the same configs but with **labeled test splits** that
power the official BalkanBench scoring pipeline. The public repo's test
splits are input-only; the labels live here.

## Why this is gated

The benchmark contract is: anyone can train and tune on labeled
train/validation, but **nobody trains against the test labels**. Keeping
test labels in a gated repo, behind an audit-logged token, lets us:

1. Score submitted predictions against canonical labels.
2. Keep the test split out of any pretraining or fine-tuning corpus.
3. Detect if a model has been overfit to test by re-running on a fresh
   sample.

The trusted scoring path uses an `HF_TOKEN` with read access to this
repo, runs `balkanbench score` against a submitted `predictions.jsonl`,
and writes the resulting `result.json` to the leaderboard.

## How model authors use this

You don't. The flow for getting on the leaderboard is:

1. `pip install balkanbench` (or use the Docker image / Vertex AI launcher).
2. `balkanbench predict --model <yours> --task <task> --language <mne>` -
   loads the **public** repo's test inputs, generates `predictions.jsonl`.
3. Submit the predictions package via the [submission flow](https://balkanbench.com).
4. Official scoring runs `balkanbench score` in the trusted environment
   (which is the only place with access to this repo) and publishes the
   row to <https://balkanbench.com/leaderboard>.

## What's inside

Same configs as the public Montenegrin preview, with labeled test splits.

## Loading (trusted scoring environment only)

```python
import os
from datasets import load_dataset

ds = load_dataset(
    "permitt/superglue-mne-private",
    "boolq",
    split="test",
    revision="v0.1.0-data",
    token=os.environ["HF_TOKEN"],
)
```

## License

CC-BY-4.0 on the data, same as the public repo. Access control is policy,
not licensing.

## Links

- **Public sibling (start here)**: <https://huggingface.co/datasets/permitt/superglue-mne>
- Live leaderboard: <https://balkanbench.com/leaderboard>
- Project home: <https://balkanbench.com>
- Background: [Release of BalkanBench - the vision behind it (Medium)](https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411)
- Source: <https://github.com/permitt/balkan-bench>
