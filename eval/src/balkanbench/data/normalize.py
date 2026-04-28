"""Pure-function dataset transformations used by the publisher.

Each function takes a ``datasets.DatasetDict`` and returns a new ``DatasetDict``;
none of them mutate the input in place and none of them call out to
HuggingFace.  Keeps the transformations unit-testable without network.
"""

from __future__ import annotations

from typing import cast

from datasets import Dataset, DatasetDict


def rename_splits(mapping: dict[str, str], dataset: DatasetDict) -> DatasetDict:
    """Rename splits according to ``mapping`` (old -> new).

    Raises ``ValueError`` if a target split name already exists on the dataset
    for a different source split (would silently drop data).
    """
    new: dict[str, Dataset] = {}
    renamed_from: dict[str, str] = {}
    for split_name, ds in dataset.items():
        target = mapping.get(split_name, split_name)
        if target in new:
            prior = renamed_from.get(target, target)
            raise ValueError(
                f"split name collision: both {prior!r} and {split_name!r} would become {target!r}"
            )
        new[target] = ds
        renamed_from[target] = split_name
    return DatasetDict(new)


def strip_label_columns(
    dataset: DatasetDict,
    *,
    split: str,
    label_fields: list[str],
) -> DatasetDict:
    """Return ``dataset`` with ``label_fields`` removed from the named ``split``.

    Missing columns are silently ignored (no-op): callers should not care whether
    the upstream data already stripped labels or not.
    """
    if split not in dataset:
        return dataset
    current = dataset[split]
    present = [col for col in label_fields if col in current.column_names]
    if not present:
        return dataset
    stripped = current.remove_columns(present)
    out = {name: ds for name, ds in dataset.items()}
    out[split] = stripped
    return DatasetDict(out)


def attach_task_metadata(
    dataset: DatasetDict,
    *,
    task_id: str,
    language: str,
) -> DatasetDict:
    """Ensure every row carries ``task_id``, ``language``, and ``example_id``.

    - ``task_id`` and ``language`` are the canonical values derived from the
      caller's benchmark config; any preexisting columns are **overwritten**
      with these values so upstream drift cannot silently ship wrong IDs.
    - ``example_id`` is a stable key used later to match predictions to
      private labels. If upstream provides one, it is preserved; otherwise a
      stable ``{split}-{i}`` identifier is generated.
    """
    out: dict[str, Dataset] = {}
    for split_name, ds in dataset.items():
        num_rows = ds.num_rows

        working = ds
        if "task_id" in working.column_names:
            working = working.remove_columns(["task_id"])
        working = working.add_column("task_id", [task_id] * num_rows)

        if "language" in working.column_names:
            working = working.remove_columns(["language"])
        working = working.add_column("language", [language] * num_rows)

        if "example_id" not in working.column_names:
            working = working.add_column(
                "example_id", [f"{split_name}-{i}" for i in range(num_rows)]
            )

        out[split_name] = cast(Dataset, working)
    return DatasetDict(out)
