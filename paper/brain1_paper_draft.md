# brain-1: A Clean-Room Multimodal Benchmark for Brain Encoding

## Title

brain-1: A clean-room multimodal benchmark for predicting parcel-level fMRI responses from language, sound, and video

## Authors

Advait Paliwal

## Status

Working draft based on the current `brain-1` repository state. This draft only includes claims supported by the runs and artifacts currently present in the repo.

## Abstract

We present `brain-1`, a clean-room benchmark for multimodal brain encoding inspired by recent foundation-model approaches to fMRI prediction. Rather than extending a non-commercial research release directly, `brain-1` rebuilds the end-to-end pipeline around commercially usable components and explicit held-out evaluation. The current system predicts Schaefer-1000 parcel responses from language, audio, and video-derived features using a subject-aware temporal regression model with hemodynamic smoothing. On the Algonauts 2025 / CNeuroMod dataset family, our best text-only model, trained on `Friends` seasons 1-5 and evaluated on held-out season 6 across four subjects, achieves a held-out Pearson correlation of `0.1053`. We further establish working audio and video branches on real movie stimuli and compare unimodal and multimodal variants on small held-out subsets. These experiments show that text is currently the strongest branch, audio is competitive on small held-out slices, and naive multimodal feature fusion does not yet outperform the best unimodal encoder. The current results establish a reproducible clean-room baseline and identify visual representation quality and multimodal alignment as the dominant bottlenecks.

## 1. Introduction

Foundation-model-based brain encoding has become a practical route to predicting neural responses from naturalistic stimuli. Recent work such as TRIBE v2 demonstrates that large pretrained audio, language, and visual models can be mapped to fMRI with strong zero-shot and held-out performance. However, the public TRIBE v2 codebase is released under a non-commercial license and is coupled to a specific research stack. If the goal is a commercially usable or independently extensible brain-encoding model, a clean-room benchmark is required.

`brain-1` is intended as that benchmark. The project keeps the core scientific framing of modern brain encoding systems:

- align naturalistic stimuli to fMRI time grids
- extract modality-specific latent representations
- predict parcel-level responses with subject-aware temporal models
- validate on held-out stimuli rather than random row splits

At the same time, `brain-1` intentionally separates itself from TRIBE’s exact code path, pretrained stack, and serving assumptions. The current repository focuses on establishing a transparent benchmark with reproducible artifacts and explicit tradeoff testing across encoders.

## 2. Related Work

The immediate reference point is TRIBE v2, which uses pretrained language, audio, and video backbones to predict human brain activity from naturalistic and experimental stimuli. The local copy of the TRIBE v2 paper at `/Users/advaitpaliwal/Downloads/tribev2.pdf` describes a tri-modal system trained on over 1,000 hours of fMRI across 720 subjects and reports state-of-the-art performance on the Algonauts 2025 challenge.

`brain-1` should not be interpreted as a direct apples-to-apples replacement for TRIBE v2. It is a smaller clean-room benchmark on the same dataset family, using held-out validation and commercially usable components, with the explicit goal of identifying which branches and design choices are worth scaling.

## 3. Data

### 3.1 Primary dataset family

The current benchmark uses the public Algonauts 2025 / CNeuroMod release:

- subjects: `sub-01`, `sub-02`, `sub-03`, `sub-05`
- target space: Schaefer-1000 parcels
- stimuli: `Friends` episodes and splits

The repository contains ingestion scripts that operate directly over the DataLad dataset layout:

- `scripts/download_algonauts2025.py`
- `scripts/build_algonauts_text_manifest.py`
- `scripts/build_algonauts_video_manifest.py`
- `scripts/download_algonauts_videos.py`

### 3.2 Current evaluation split

The strongest current benchmark uses:

- train: `Friends` seasons `1-5`
- validation: held-out `Friends` season `6`

This split is intentionally stimulus-level, not a random row split, so it tests generalization to unseen episodes.

## 4. Model

### 4.1 Core prediction head

The current parcel prediction head in `brain-1` has four core components:

1. Modality feature projector
2. Subject embedding and subject bias projection
3. Temporal transformer adapter
4. HRF-inspired temporal smoothing followed by parcel regression

The single-modality version is implemented in:

- `src/brain_1/models/brain_model.py`

The modality-aware multimodal version is implemented in:

- `src/brain_1/models/multimodal_brain_model.py`

### 4.2 Encoders

The current repository has exercised the following encoder branches:

- Text:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen3-Embedding-0.6B`
- Audio:
  - `facebook/w2v-bert-2.0`
- Video:
  - `google/siglip-base-patch16-224`

A stronger video path using `Qwen2.5-VL` has been scaffolded but is not yet benchmarked successfully.

### 4.3 Temporal alignment

The benchmark uses time-aligned features at the fMRI-response scale:

- text rows are derived from transcript chunks already aligned to TR-like intervals
- video is sampled as TR-aligned frames
- audio is sampled as TR-aligned WAV clips

For small held-out multimodal experiments, all features are truncated to the shared minimum sequence length before training.

## 5. Training

### 5.1 Objective

The training objective is masked mean squared error over parcel targets. Evaluation additionally reports a flattened Pearson correlation over held-out valid positions. These metrics are useful for comparing `brain-1` runs internally, though they are not yet a full parcel-wise evaluation suite in the style of TRIBE.

### 5.2 Checkpointing

The repo saves:

- `best.pt` during validation-aware runs
- `final.pt` at the end of training
- `history.json` where applicable

Because MPS checkpoint writes can be fragile with default PyTorch zip serialization, the repository uses CPU state dict conversion and a safer checkpoint save path.

## 6. Results

### 6.1 Main held-out benchmark

The strongest current result is:

- run: `artifacts/algonauts_text_baseline_s1_s5_all4_tuned_b2/best.pt`
- train: seasons `1-5`
- validation: season `6`
- subjects: `4`
- held-out rows: `199`

Metrics:

```json
{
  "mse": 0.3683369755744934,
  "pearson": 0.10527008771896362,
  "loss": 0.36837509214878084,
  "num_rows": 199
}
```

Detailed evaluation of the same checkpoint yields:

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

### 6.2 Encoder comparison on the same split

Current summary from `artifacts/benchmark_summary.json`:

```json
{
  "best_run": "qwen25_s1s5_tuned_b2",
  "runs": {
    "qwen25_s1s5_final": {
      "mse": 0.36853334307670593,
      "pearson": 0.09849394112825394
    },
    "qwen25_s1s5_tuned_b2": {
      "mse": 0.3683369755744934,
      "pearson": 0.10527008771896362
    },
    "qwen3_s1s5": {
      "mse": 0.37174439430236816,
      "pearson": 0.011148706078529358
    }
  }
}
```

The main takeaway is that `Qwen3-Embedding-0.6B` underperformed the stronger `Qwen2.5-0.5B-Instruct` text feature baseline on this held-out fMRI benchmark.

### 6.3 Small-slice multimodal comparisons

On a 4-stimulus held-out movie split (`s01e01a`, `s01e01b`, `s01e02a`, `s01e02b`):

```json
{
  "text_quad": {
    "pearson": 0.10050234943628311
  },
  "audio_quad": {
    "pearson": 0.09786557406187057
  },
  "video_quad": {
    "pearson": 0.059796132147312164
  },
  "text_audio_concat_quad": {
    "pearson": 0.040300238877534866
  },
  "text_audio_fusion_quad": {
    "pearson": 0.08054832369089127
  },
  "text_video_concat_quad": {
    "pearson": 0.042754460126161575
  },
  "text_video_fusion_quad": {
    "pearson": 0.037360936403274536
  }
}
```

This is not a final multimodal verdict, but it is already informative:

- text is currently the strongest branch
- audio is nearly as strong as text on small held-out slices
- video works, but is weaker than text and audio with the current SigLIP-based setup
- naive concatenation is not enough to get multimodal gains

### 6.4 Four-stimulus modality comparison

We further benchmarked unimodal and multimodal variants on a held-out split formed from four real
movie segments from season 1 of `Friends`, training on the `a` segments and evaluating on the held-out
`b` segments across the four public Algonauts subjects.

Results:

| Model | Held-out Pearson | Held-out MSE |
| --- | ---: | ---: |
| Text only | 0.1005 | 0.3713 |
| Audio only | 0.0979 | 0.3909 |
| Video only | 0.0598 | 0.3927 |
| Text + Audio (concat) | 0.0403 | 0.3933 |
| Text + Audio (fusion) | 0.0805 | 0.3915 |
| Text + Video (concat) | 0.0428 | 0.3944 |
| Text + Video (fusion) | 0.0374 | 0.4550 |
| Text + Audio + Video (trimodal) | 0.0679 | 0.3921 |

The main conclusion from this table is that the current bottleneck lies in multimodal fusion and
visual representation quality rather than in the parcel regression head itself. Audio already carries
predictive signal nearly as strong as text on this held-out slice, while adding video with the current
encoder and fusion stack degrades performance rather than improving it.

### 6.5 Detailed subject and parcel evaluation

For the current best safe checkpoint on the `s1-s5 -> s6` benchmark, we also computed more
TRIBE-like detailed metrics:

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

This confirms that the benchmark signal is not driven by a single subject and that the current model
produces stable but still modest parcel-level predictivity on unseen season-6 stimuli.

## 7. Discussion

The current `brain-1` results suggest that the main bottleneck is not the regression head. The bottleneck is representation quality and multimodal alignment, especially for video. The text branch is already competitive enough to produce stable held-out signal on season-level splits. Audio is promising. Video is functioning, but still underpowered relative to language and sound. That pattern implies that the next meaningful gain will likely come from a stronger video encoder or a better cross-modal integration strategy rather than from additional MLP or transformer depth in the parcel head.

## 8. Limitations

- The current evaluation metric is a flattened Pearson rather than a full parcel-wise analysis suite.
- The current multimodal comparisons are still small-slice sanity checks.
- The stronger visual branch (`Qwen2.5-VL`) is scaffolded but not yet benchmarked to completion.
- The current work remains a benchmark, not a finished multimodal foundation model.

## 9. Conclusion

`brain-1` now constitutes a real clean-room benchmark on the Algonauts 2025 family with:

- reproducible train/validation splits
- real held-out checkpoint selection
- working text, audio, and video branches
- early multimodal ablations

The strongest current result remains a text-only model, but the audio branch is close and the video branch is operational. The path forward is clear: strengthen visual representations, improve cross-modal fusion, and then scale the best-performing configuration to longer-running training infrastructure.
