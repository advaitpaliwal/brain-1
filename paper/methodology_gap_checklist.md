# Methodology Gap Checklist

This file tracks the gap between the current `brain-1` benchmark and a TRIBE-style paper reproduction.

## Done

- Clean-room codebase
- Held-out stimulus evaluation on Algonauts / CNeuroMod
- Text branch benchmark
- Audio branch benchmark
- Video branch benchmark
- Pairwise multimodal ablations
- Trimodal ablation on a small held-out slice
- Parcel-wise and subject-wise evaluation for a main held-out checkpoint
- Paper draft with real benchmark numbers

## In Progress

- Stronger visual encoder benchmark (`Qwen2.5-VL`)
- Real GPU benchmark on Modal using processed feature tensors

## Missing for a fuller TRIBE-style reproduction

- Full dataset mix beyond Algonauts:
  - BOLD Moments
  - Lebel2023
  - Wen2017
- More faithful training schedule:
  - explicit warmup + cosine decay in all main runs
  - early stopping on validation Pearson as standard
- Cleaner multimodal alignment:
  - consistent 2 Hz stimulus grid
  - explicit 5 second hemodynamic offset handling in every branch
- Statistical reporting closer to the paper:
  - parcel distribution summaries
  - subject-wise significance / confidence intervals
  - ablation figures instead of only JSON metrics
- Better multimodal fusion:
  - current fusion underperforms strongest unimodal models

## Current best benchmark to build from

- Train: `Friends s1-s5`
- Validation: `Friends s6`
- Subjects: `sub-01`, `sub-02`, `sub-03`, `sub-05`
- Best held-out checkpoint:
  - `artifacts/algonauts_text_baseline_s1_s5_all4_tuned_b2/best.pt`
