# TRIBE v2 vs brain-1

This file records what is and is not currently comparable between the public TRIBE v2 release and this `brain-1` clean-room benchmark.

## What TRIBE v2 claims in the local paper

From the local paper copy at `/Users/advaitpaliwal/Downloads/tribev2.pdf`:

- TRIBE v2 is a tri-modal model for `video`, `audio`, and `language`.
- It is trained on `over 1,000 hours of fMRI` across `720 subjects`.
- It reports first place performance on the `Algonauts 2025` competition.
- In the local paper's competition table, the reported top score is `0.2146 ± 0.0312`.
- It is evaluated across multiple datasets, not just one narrow train/val split.

## What brain-1 currently is

`brain-1` is currently a much smaller clean-room benchmark:

- dataset family: `Algonauts 2025`
- current strongest held-out benchmark:
  - train: `Friends seasons 1-5`
  - validation: `Friends season 6`
  - subjects: `sub-01`, `sub-02`, `sub-03`, `sub-05`
  - best held-out Pearson: `0.10527008771896362`

Best checkpoint:

```text
artifacts/algonauts_text_baseline_s1_s5_all4_tuned_b2/best.pt
```

## What is directly comparable

- Both systems target the same broad scientific problem: predicting fMRI responses from naturalistic stimuli.
- Both use the `Algonauts 2025 / CNeuroMod` data family.
- Both evaluate with parcel-level Pearson-style encoding metrics.

## What is not directly comparable

- TRIBE v2 is trained on far more data and more modalities.
- The public TRIBE demo checkpoint is not the same training/eval setup as the current `brain-1` split.
- `brain-1` currently uses a clean-room training stack and different encoders.
- `brain-1` is still mostly a text-first benchmark with small video/audio slices added for branch validation.

## Current conclusion

- `brain-1` is not yet a fair apples-to-apples replacement for TRIBE v2.
- It is a valid clean-room baseline on the same dataset family.
- On the current held-out benchmark, `brain-1` (`0.1053`) is still well below the local-paper TRIBE competition score (`0.2146 ± 0.0312`), though these are not matched evaluation protocols.
- The strongest current `brain-1` result is still text-only.
- Audio is competitive on small held-out slices.
- Video is working, but weaker than text and audio with the current visual encoder.
