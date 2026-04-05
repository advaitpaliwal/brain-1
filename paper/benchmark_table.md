# Benchmark Table

## Main held-out benchmark

| Run | Train split | Validation split | Metric |
| --- | --- | --- | ---: |
| Qwen2.5 text (final) | Friends s1-s5 | Friends s6 | 0.0985 |
| Qwen2.5 text (tuned b2) | Friends s1-s5 | Friends s6 | 0.1053 |
| Qwen3 embedding text | Friends s1-s5 | Friends s6 | 0.0111 |

## 4-stimulus held-out benchmark

| Run | Pearson | MSE |
| --- | ---: | ---: |
| Text | 0.1005 | 0.3713 |
| Audio | 0.0979 | 0.3909 |
| Video | 0.0598 | 0.3927 |
| Text + Audio concat | 0.0403 | 0.3933 |
| Text + Audio fusion | 0.0805 | 0.3915 |
| Text + Video concat | 0.0428 | 0.3944 |
| Text + Video fusion | 0.0374 | 0.4550 |
| Text + Audio + Video | 0.0679 | 0.3921 |

## Detailed evaluation for the best safe checkpoint

| Metric | Value |
| --- | ---: |
| Mean parcel Pearson | 0.0851 |
| Mean subject Pearson | 0.1069 |
| Subject `sub-01` | 0.1005 |
| Subject `sub-02` | 0.1067 |
| Subject `sub-03` | 0.1156 |
| Subject `sub-05` | 0.1048 |
