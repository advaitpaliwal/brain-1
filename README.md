# brain-1

`brain-1` is a clean-room commercial brain-encoding scaffold inspired by TRIBE-like multimodal fMRI prediction, but implemented as a fresh project.

## V1 scope

- Predict `1,000` Schaefer parcels first, not dense cortical vertices.
- Use one commercial-friendly multimodal backbone as the feature extractor.
- Align text, audio, and video on a shared temporal grid.
- Train a small HRF-aware brain head before touching the backbone heavily.

## Initial architecture

1. Input datasets
- `Algonauts 2025`
- `BOLD Moments`
- `Lebel 2023`

2. Backbone
- Multimodal feature extractor interface.
- Default planned implementation: `Qwen2.5-Omni-7B`.

3. Temporal brain head
- Modality projection
- Temporal adapter
- HRF smoothing
- Parcel regression head

4. Optional later heads
- Vertex upsampler
- ROI summary head
- Subject adaptation layers

## Repo layout

```text
brain-1/
  configs/
  scripts/
  src/brain_1/
  tests/
```

## Quick start

```bash
cd brain-1
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m py_compile $(rg --files . -g '*.py')
```

Create a dummy dataset and run the first smoke-train:

```bash
python scripts/create_dummy_dataset.py --output data/manifests/train.jsonl
python scripts/train.py
```

Fast verification without any data files:

```bash
PYTHONPATH=src python scripts/train.py \
  --train-config configs/smoke_train.yaml \
  --model-config configs/smoke_model.yaml
```

## First real baseline: Algonauts 2025 text

Install the minimal dataset slice:

```bash
source .venv/bin/activate
python scripts/download_algonauts2025.py \
  --output data/raw/algonauts2025 \
  --subject sub-01 \
  --season 1
```

Build a raw manifest from transcript segments and parcel targets:

```bash
python scripts/build_algonauts_text_manifest.py \
  --dataset-root data/raw/algonauts2025 \
  --subject sub-01 \
  --season 1 \
  --limit 2 \
  --output data/manifests/algonauts2025_text_raw.jsonl
```

Extract Qwen text features for the transcript chunks:

```bash
python scripts/extract_algonauts_text_features.py \
  --raw-manifest data/manifests/algonauts2025_text_raw.jsonl \
  --output-manifest data/manifests/algonauts2025_text_train.jsonl \
  --feature-root data/processed/algonauts2025_text \
  --limit 2 \
  --device cpu
```

Train the parcel head on the processed manifest:

```bash
PYTHONPATH=src python scripts/train.py \
  --train-config configs/algonauts_text_train.yaml \
  --model-config configs/model.yaml
```

The baseline checkpoint is saved to:

```text
artifacts/algonauts_text_baseline/final.pt
```

## Stronger local baseline: full season 1, all 4 Algonauts subjects

Build the full season-1 raw manifest:

```bash
python scripts/build_algonauts_text_manifest.py \
  --dataset-root data/raw/algonauts2025 \
  --subjects sub-01,sub-02,sub-03,sub-05 \
  --season 1 \
  --output data/manifests/algonauts2025_text_raw_full_s1_all4.jsonl
```

Materialize shared stimulus features and per-subject targets:

```bash
python scripts/extract_algonauts_text_features.py \
  --raw-manifest data/manifests/algonauts2025_text_raw_full_s1_all4.jsonl \
  --output-manifest data/manifests/algonauts2025_text_train_full_s1_all4.jsonl \
  --feature-root data/processed/algonauts2025_text_shared_s1_sub01_sub02 \
  --batch-size 64 \
  --device mps \
  --share-features-across-subjects
```

Train the all-four-subject local baseline:

```bash
PYTHONPATH=src python scripts/train.py \
  --train-config configs/algonauts_text_train_full_s1_all4.yaml \
  --model-config configs/model.yaml
```

The full local checkpoint is saved to:

```text
artifacts/algonauts_text_baseline_full_s1_all4/final.pt
```

## Held-out validation baseline

Split the full season-1 all-subject manifest by held-out stimuli:

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/algonauts2025_text_train_full_s1_all4.jsonl \
  --train-output data/manifests/algonauts2025_text_train_full_s1_all4_train.jsonl \
  --val-output data/manifests/algonauts2025_text_train_full_s1_all4_val.jsonl \
  --val-regex 's01e(21|22|23|24)[ab]$'
```

Train on the train split:

```bash
PYTHONPATH=src python scripts/train.py \
  --train-config configs/algonauts_text_train_full_s1_all4_split.yaml \
  --model-config configs/model.yaml
```

Evaluate on held-out validation stimuli:

```bash
PYTHONPATH=src python scripts/evaluate_manifest.py \
  --checkpoint artifacts/algonauts_text_baseline_full_s1_all4_split/final.pt \
  --manifest data/manifests/algonauts2025_text_train_full_s1_all4_val.jsonl \
  --model-config configs/model.yaml \
  --output artifacts/algonauts_text_baseline_full_s1_all4_split/val_metrics.json \
  --device mps \
  --batch-size 2
```

Current held-out validation metrics:

```json
{
  "mse": 0.3816055655479431,
  "pearson": 0.08518872410058975,
  "loss": 0.38161194510757923,
  "num_rows": 32
}
```

## Stronger benchmark: seasons 1-5 train, season 6 validation

Build the all-seasons raw manifest:

```bash
python scripts/build_algonauts_text_manifest.py \
  --dataset-root data/raw/algonauts2025 \
  --subjects sub-01,sub-02,sub-03,sub-05 \
  --seasons 1-6 \
  --output data/manifests/algonauts2025_text_raw_s1_s6_all4.jsonl
```

Materialize shared text features and per-subject targets:

```bash
python scripts/extract_algonauts_text_features.py \
  --raw-manifest data/manifests/algonauts2025_text_raw_s1_s6_all4.jsonl \
  --output-manifest data/manifests/algonauts2025_text_train_s1_s6_all4.jsonl \
  --feature-root data/processed/algonauts2025_text_shared_s1_sub01_sub02 \
  --batch-size 64 \
  --device mps \
  --share-features-across-subjects
```

Split seasons `1-5` for training and season `6` for validation:

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/algonauts2025_text_train_s1_s6_all4.jsonl \
  --train-output data/manifests/algonauts2025_text_train_s1_s5_all4.jsonl \
  --val-output data/manifests/algonauts2025_text_val_s6_all4.jsonl \
  --val-regex '^s06e'
```

Train on seasons `1-5`:

```bash
PYTHONPATH=src python scripts/train.py \
  --train-config configs/algonauts_text_train_s1_s5_all4.yaml \
  --model-config configs/model.yaml
```

Evaluate on held-out season `6`:

```bash
PYTHONPATH=src python scripts/evaluate_manifest.py \
  --checkpoint artifacts/algonauts_text_baseline_s1_s5_all4/final.pt \
  --manifest data/manifests/algonauts2025_text_val_s6_all4.jsonl \
  --model-config configs/model.yaml \
  --output artifacts/algonauts_text_baseline_s1_s5_all4/val_metrics.json \
  --device mps \
  --batch-size 2
```

Current season-6 held-out validation metrics:

```json
{
  "mse": 0.36853334307670593,
  "pearson": 0.09849394112825394,
  "loss": 0.36857557684183123,
  "num_rows": 199
}
```

## Current best text-only run

The current best held-out text-only checkpoint is:

```text
artifacts/algonauts_text_baseline_s1_s5_all4_tuned_b2/best.pt
```

Its season-6 held-out validation metrics are:

```json
{
  "mse": 0.3683369755744934,
  "pearson": 0.10527008771896362,
  "loss": 0.36837509214878084,
  "num_rows": 199
}
```

The current benchmark summary is in:

```text
artifacts/benchmark_summary.json
```

This also records that the `Qwen3-Embedding-0.6B` comparison underperformed the stronger
`Qwen2.5-0.5B-Instruct` feature baseline on the same `s1-s5 -> s6` split.

## Tiny multimodal sanity check

Using two real `Friends` movie segments (`s01e01a` train, `s01e01b` validation), all four
subjects, and TR-aligned sampled video frames:

- text-only tiny best Pearson: `0.0856`
- video-only tiny best Pearson: `0.0366`
- text+video tiny best Pearson: `0.0263`

Summary file:

```text
artifacts/tiny_multimodal_summary.json
```

Current takeaway:

- the text branch is already materially stronger than the tiny video branch
- naive feature concatenation did not help on the tiny split
- the next meaningful multimodal improvement will need a better visual encoder or a better
  fusion strategy, not just stacking two pretrained feature streams together

## 4-stimulus text/video benchmark

Using four real `Friends` season-1 segments (`s01e01a`, `s01e01b`, `s01e02a`, `s01e02b`) with
train on the `a` segments and validation on the held-out `b` segments:

- text-only best Pearson: `0.1005`
- video-only best Pearson: `0.0598`
- text+video naive concat best Pearson: `0.0428`
- text+video modality-aware fusion best Pearson: `0.0374`

Summary file:

```text
artifacts/quad_multimodal_summary.json
```

Current takeaway:

- text remains the strongest branch on the larger held-out slice
- video is now a real, working branch and is nontrivial, but still behind text
- current multimodal fusion is not yet good enough; the next win is likely a better video
  encoder or better cross-modal fusion, not more of the same projector stack

## 4-stimulus modality benchmark with audio

Using four real `Friends` clips (`s01e01a`, `s01e01b`, `s01e02a`, `s01e02b`) with train on the
`a` clips and validation on the held-out `b` clips:

- text-only best Pearson: `0.1005`
- audio-only best Pearson: `0.0979`
- video-only best Pearson: `0.0598`
- text+video concat best Pearson: `0.0428`
- text+video fusion best Pearson: `0.0374`
- text+audio+video trimodal best Pearson: `0.0726`

Summary file:

```text
artifacts/quad_multimodal_extended_summary.json
```

Current takeaway:

- text is still the single best branch on this split
- audio is surprisingly close and currently much stronger than video
- the current multimodal fusion stack is not yet beating the strongest unimodal branch
- the next best investment is likely stronger visual features and better trimodal alignment

## 4-stimulus text/video/audio benchmark

Using four real `Friends` clips (`s01e01a`, `s01e01b`, `s01e02a`, `s01e02b`) with train on the
`a` clips and validation on the held-out `b` clips:

- text-only best Pearson: `0.1005`
- audio-only best Pearson: `0.0979`
- video-only best Pearson: `0.0598`
- text+audio concat best Pearson: `0.0403`
- text+audio fusion best Pearson: `0.0805`
- text+video concat best Pearson: `0.0428`
- text+video fusion best Pearson: `0.0374`

Summary file:

```text
artifacts/quad_multimodal_extended_summary.json
```

Current takeaway:

- text and audio are both strong on this held-out slice
- video is meaningfully weaker than either text or audio right now
- the current multimodal fusion stack is not yet beating the strongest unimodal branch
- the next real gain is likely better multimodal alignment/fusion or stronger visual features, not more
  plain concatenation

## Current best checkpoint

Across the real held-out benchmarks so far, the current best checkpoint is still:

```text
artifacts/algonauts_text_baseline_s1_s5_all4_tuned_b2/best.pt
```

with held-out season-6 validation:

```json
{
  "mse": 0.3683369755744934,
  "pearson": 0.10527008771896362,
  "loss": 0.36837509214878084,
  "num_rows": 199
}
```

Detailed held-out metrics for the same checkpoint:

```json
{
  "loss": 0.36763607084751126,
  "mean_parcel_pearson": 0.08506268232068397,
  "mean_subject_pearson": 0.10692047700285912,
  "subject_scores": {
    "sub-01": 0.10051405429840088,
    "sub-02": 0.10669812560081482,
    "sub-03": 0.11564770340919495,
    "sub-05": 0.10482202470302582
  },
  "num_parcels": 1000,
  "num_rows": 199
}
```

Comparison files:

```text
artifacts/benchmark_summary.json
artifacts/quad_multimodal_extended_summary.json
TRIBE_COMPARISON.md
```

## Immediate next steps

1. Replace the current SigLIP video features with a stronger vision-language encoder path such as `Qwen2.5-VL`.
2. Push the audio branch through the same held-out comparison loops used for text and video.
3. Upgrade multimodal fusion beyond plain projector stacking so it can beat the strongest unimodal branch.
4. Move the longer extraction and benchmark runs to Modal once the local pipelines are stable.
