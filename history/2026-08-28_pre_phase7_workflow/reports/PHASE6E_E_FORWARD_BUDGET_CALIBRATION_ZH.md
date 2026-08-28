# Phase 6E-E Same Forward-Budget Calibration

Updated: 2026-06-23

## Purpose

This note recalibrates the current masked-diffusion/direct comparison using a
hardware-stable neural forward-equivalent budget rather than wall-clock
time-matching. It uses only the frozen local Stage 3.8 three-seed evidence:

- `artifacts/phase6e-e-stage38-sealed-evaluation/final_evidence.json`
- `artifacts/phase6e-e-stage38-rawonly-evaluation/rawonly_evidence.json`
- per-instance records under `artifacts/phase6e-e-stage38-sealed-evaluation/records/`

No new training, tuning, or data opening is involved.

## Cost Definition

Use the neural forward-equivalent budget

```text
B_NN = N_prop * N_step
```

where `N_prop` is the number of generated proposals and `N_step` is the number
of typed graph neural forward evaluations required to produce one proposal.
For the direct predictor, `N_step = 1`. For absorbing-MASK diffusion,
`N_step` is the number of reverse conditional completion steps.

The current hard-cost anchor is:

| Method | Proposals | Reverse/forward steps | B_NN |
| --- | ---: | ---: | ---: |
| Direct K=64 | 64 | 1 | 64 |
| Masked Diffusion K=8 | 8 | 8 | 64 |

`Masked deterministic K=1` has `B_NN = 8` and should be treated as a mechanism
ablation for conditional completion rather than the equal-budget competitor to
Direct K=64.

## Frozen Three-Seed Results Under B_NN = 64

| Layer | Metric | Direct K=64 | Masked Diffusion K=8 | Reading |
| --- | --- | ---: | ---: | --- |
| Raw-only | Raw success | 48.96% | 95.57% | Strong masked feasibility advantage |
| Raw-only | Raw feasible proposal rate | 34.65% | 94.27% | Strong masked proposal-quality advantage |
| Raw-only | Raw gap | 4.009% | 2.985% | 25.54% aggregate gap reduction |
| Pre-fallback | Success | 98.18% | 98.18% | Feasibility tie after bounded repair |
| Pre-fallback | Gap | 3.693% | 2.912% | 21.15% aggregate gap reduction |
| Final pipeline | Success | 100.00% | 100.00% | Final feasibility is a pipeline property |
| Final pipeline | Gap | 1.631% | 1.254% | Favorable aggregate trend |
| Final pipeline | Total time | 0.501 s | 0.486 s | Hardware-specific, secondary evidence |

Raw-only paired comparison between Direct K=64 and Masked Diffusion K=8:

- masked wins: 110
- direct wins: 12
- ties: 6
- sign-test p-value: `5.47e-21`

Pre-fallback paired comparison from the Stage 3.8 gate:

- masked wins: 70
- direct wins: 51
- ties: 7
- sign-test p-value: `0.101`

## Seed-Level Check

| Seed | Raw success gain | Pre-fallback gap change | Final gap change | Time ratio |
| --- | ---: | ---: | ---: | ---: |
| 2026070111 | +50.00 pp | +48.42% | +32.90% | 0.956 |
| 2026070112 | +42.97 pp | -19.99% | -3.61% | 0.949 |
| 2026070113 | +46.88 pp | +28.51% | +36.77% | 1.006 |

The raw feasibility improvement is stable across all three seeds. Latency-gap
improvement is not stable across all seeds because seed `2026070112` remains
adverse in pre-fallback and final gap.

## Selected-Source Breakdown

| Method | Selected raw | Selected repair | Selected fallback |
| --- | ---: | ---: | ---: |
| Direct K=64 | 7.03% | 27.34% | 65.62% |
| Masked deterministic K=1 | 31.25% | 1.56% | 67.19% |
| Masked Diffusion K=8 | 36.72% | 2.08% | 61.20% |

This supports the interpretation that masked diffusion improves the raw
candidate pool and reduces the need for repair relative to Direct K=64, while
the always-available fallback still accounts for a large fraction of final
accepted outputs.

## Calibrated Claim Standard

Recommended claim hierarchy for Section V:

1. **Primary hard-cost model claim.** Under the same neural
   forward-equivalent budget (`B_NN = 64`), absorbing-MASK diffusion strongly
   improves raw neural proposal feasibility and aggregate raw proposal quality
   over the direct predictor.
2. **Pipeline claim.** After shared bounded repair, fallback, and exact final
   selection, masked diffusion shows a favorable aggregate full-pipeline trend,
   but the current three-seed sealed gate does not confirm end-to-end dominance.
3. **Efficiency claim.** Wall-clock time should be reported, but it is
   hardware-specific and should not be the sole fairness metric.

Avoid:

- claiming statistically confirmed full-pipeline superiority from Stage 3.8;
- using final 100% success as neural feasibility evidence;
- treating wall-clock time matching as a hardware-independent standard.

Next calibration step: when Stage 3.9 ten-seed training and evaluation are
complete, recompute this exact forward-budget table over all ten seeds. A
stronger final claim requires the pre-fallback/final paired statistics and
seed-level behavior to improve, not merely the raw feasibility layer.
