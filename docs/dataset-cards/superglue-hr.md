---
license: cc-by-4.0
language:
  - hr
multilinguality: monolingual
pretty_name: BalkanBench SuperGLUE - Croatian
size_categories:
  - 10K<n<100K
source_datasets:
  - extended|super_glue
task_categories:
  - text-classification
  - multiple-choice
  - question-answering
tags:
  - balkanbench
  - bcms
  - croatian
  - superglue
  - nlu
  - benchmark
configs:
  - config_name: boolq
    data_files: { train: boolq/train-*.parquet, validation: boolq/validation-*.parquet, test: boolq/test-*.parquet }
  - config_name: cb
    data_files: { train: cb/train-*.parquet, validation: cb/validation-*.parquet, test: cb/test-*.parquet }
  - config_name: copa
    data_files: { train: copa/train-*.parquet, validation: copa/validation-*.parquet, test: copa/test-*.parquet }
  - config_name: multirc
    data_files: { train: multirc/train-*.parquet, validation: multirc/validation-*.parquet, test: multirc/test-*.parquet }
  - config_name: rte
    data_files: { train: rte/train-*.parquet, validation: rte/validation-*.parquet, test: rte/test-*.parquet }
---

# BalkanBench SuperGLUE - Croatian

> **Part of [BalkanBench](https://balkanbench.com)** - the open, reproducible
> benchmark for language models across Serbian, Croatian, Montenegrin, and
> Bosnian (BCMS). Live leaderboard at <https://balkanbench.com/leaderboard>.
> Background: [Release of BalkanBench - the vision behind it
> (Medium, 2026-04-27)](https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411).

This is the **Croatian SuperGLUE** released preview track of BalkanBench v0.1.
Croatian publishes alongside Serbian (the official frozen track) and
Montenegrin as a 5-task preview: same scoring contract, same private test
labels, no WSC adaptation yet.

## What's inside

5 task configs, full train + validation + public test splits (test-input
visible, **test labels hidden**):

| Config  | What it tests                                            | Train | Validation | Test  |
|---------|----------------------------------------------------------|------:|-----------:|------:|
| boolq   | Yes/no question answering over passages                  | 9,427 | 3,270      | 3,245 |
| cb      | 3-way textual entailment (entail / contradict / neutral) |   250 | 56         |   250 |
| copa    | Causal reasoning between two alternatives                |   400 | 100        |   500 |
| multirc | Multi-sentence reading comprehension (multiple correct)  | 27,243| 4,848      | 9,693 |
| rte     | Binary textual entailment                                | 2,490 | 277        | 3,000 |
| **TOTAL** |                                                        | **39,810** | **8,551** | **16,688** |

Total: **65,049 items**. WSC and the AX-b/AX-g diagnostics are
Serbian-only in v0.1 - they may be added in a later release once a
Croatian adaptation is available.

## Hidden test labels

Every config's `test` split here is **input-only**. The matching labels live
in a gated sibling repo,
[`permitt/superglue-hr-private`](https://huggingface.co/datasets/permitt/superglue-hr-private),
and are accessed only by the official scoring pipeline. To get a leaderboard
score you generate predictions on the public test inputs with `balkanbench
predict`, then submit through the [submission flow](https://balkanbench.com).

## Loading

```python
from datasets import load_dataset

boolq = load_dataset("permitt/superglue-hr", "boolq")
print(boolq["train"][0])

# All ranked tasks
for cfg in ["boolq", "cb", "copa", "rte", "multirc"]:
    ds = load_dataset("permitt/superglue-hr", cfg)
    print(cfg, {sp: len(ds[sp]) for sp in ds})
```

Pin a revision for reproducible runs:

```python
load_dataset("permitt/superglue-hr", "boolq", revision="v0.1.0-data")
```

## Schema

Each example carries an integer row id `idx` and the task-specific input
fields plus a `label` column. For `cb`:

```python
{
    "idx": 0,
    "premise": "Polly je morala brzo razmišljati. ...",
    "hypothesis": "Polly nije bila iskusna oceanska jedriličarka",
    "label": 0,  # 0=entailment, 1=contradiction, 2=neutral
}
```

Per-task field lists are in
[`eval/configs/benchmarks/superglue/tasks/`](https://github.com/permitt/balkan-bench/tree/main/eval/configs/benchmarks/superglue/tasks)
in the BalkanBench repo.

## Methodology

- **Translation**: original SuperGLUE English -> Croatian via translation +
  human verification by native speakers.
- **Frozen splits**: train/validation/test row counts and IDs pinned at
  tag `v0.1.0-data`.
- **Test labels stay private**: see the gated sibling repo above.

Full methodology, scoring contract, and provenance:
[Benchmark contract](https://github.com/permitt/balkan-bench/blob/main/docs/methodology/benchmark_contract.md)
· [Data provenance](https://github.com/permitt/balkan-bench/blob/main/docs/methodology/data_provenance.md)
· [Versioning](https://github.com/permitt/balkan-bench/blob/main/docs/methodology/versioning.md).

## Citation

If you use this dataset, please cite the Serbian SuperGLUE paper (which
introduces the BCMS adaptation methodology used here), the original
SuperGLUE paper, and the BalkanBench release:

```bibtex
@inproceedings{perovic-mihajlov-2026-serbian,
    title = "{S}erbian {S}uper{GLUE}: Towards an Evaluation Benchmark for {S}outh {S}lavic Language Models",
    author = "Perovic, Mitar and Mihajlov, Teodora",
    editor = "Hettiarachchi, Hansi and Ranasinghe, Tharindu and Plum, Alistair and
              Rayson, Paul and Mitkov, Ruslan and Gaber, Mohamed and Premasiri, Damith and
              Tan, Fiona Anting and Uyangodage, Lasitha",
    booktitle = "Proceedings of the Second Workshop on Language Models for Low-Resource Languages (LoResLM 2026)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.loreslm-1.30/",
    doi = "10.18653/v1/2026.loreslm-1.30",
    pages = "347--361",
    ISBN = "979-8-89176-377-7"
}

@inproceedings{wang2019superglue,
  title={SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
  author={Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and
          Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  booktitle={NeurIPS},
  year={2019}
}

@misc{balkanbench2026superglue_hr,
  title={BalkanBench SuperGLUE-HR: Croatian SuperGLUE for the BCMS evaluation suite},
  author={Perović, Mitar and contributors},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/permitt/superglue-hr}},
  note={Part of BalkanBench v0.1, \url{https://balkanbench.com}}
}
```

## License

CC-BY-4.0 (matches the original SuperGLUE source license).

## Sponsor

Compute for the official v0.1 evaluation is sponsored by
**[Recrewty](https://recrewty.com)**.

## Links

- Live leaderboard: <https://balkanbench.com/leaderboard>
- Project home: <https://balkanbench.com>
- Background: [Release of BalkanBench - the vision behind it (Medium)](https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411)
- Source code: <https://github.com/permitt/balkan-bench>
- Sibling tracks:
  [SuperGLUE-SR (Serbian, official)](https://huggingface.co/datasets/permitt/superglue-sr) ·
  [SuperGLUE-MNE (Montenegrin)](https://huggingface.co/datasets/permitt/superglue-mne)
- Private (gated) labels: <https://huggingface.co/datasets/permitt/superglue-hr-private>
