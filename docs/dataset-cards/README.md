# Dataset Cards

Public-facing markdown for the BalkanBench dataset repos on Hugging Face.
Each file is the full README for one HF dataset repo, including the YAML
frontmatter that the Hub renders into the side panel.

| File | HF repo |
|------|---------|
| [`superglue-sr.md`](superglue-sr.md)  | <https://huggingface.co/datasets/permitt/superglue-sr>  |
| [`superglue-hr.md`](superglue-hr.md)  | <https://huggingface.co/datasets/permitt/superglue-hr>  |
| [`superglue-mne.md`](superglue-mne.md) | <https://huggingface.co/datasets/permitt/superglue-mne> |
| [`superglue-private-template.md`](superglue-private-template.md) | template for the gated `*-private` siblings (sr / hr / mne) |

## How to update a dataset's card

The card is just a `README.md` in the root of the HF dataset repo, with
the YAML config at the top.

```bash
# clone the repo (HF git over HTTPS, requires `huggingface-cli login`)
git clone https://huggingface.co/datasets/permitt/superglue-sr
cd superglue-sr

# overwrite README.md with this version
cp ../balkan-bench/docs/dataset-cards/superglue-sr.md README.md

git add README.md
git commit -m "card: align with BalkanBench v0.1 release"
git push
```

Or upload via the HF web UI by pasting the file contents into the README
editor on the repo settings page.

## Dataset-card image (recommended)

HF dataset pages render a thumbnail in social embeds. Adding one:

1. Create a 1200x630 PNG with the BalkanBench wordmark + a flag for the
   language (sr / hr / mne).
2. Save to `dataset-card.png` in the repo root.
3. Reference it in the YAML frontmatter:

   ```yaml
   ---
   ...
   thumbnail: dataset-card.png
   ---
   ```

A simple text-on-color banner (Figma / Canva / a single `Pillow` script)
beats anything AI-generated for a benchmark dataset - it has to look
clean at 64px in a Slack/Twitter card.
