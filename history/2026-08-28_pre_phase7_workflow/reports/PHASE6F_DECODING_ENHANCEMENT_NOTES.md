# Phase 6F Neural Decoding Enhancement Notes

Date: 2026-07-02

## Motivation

The latency-aware heuristic is a strong hand-crafted constructive portfolio and
will not be kept as a primary manuscript baseline for the current learned
proposal-generator story.  The immediate goal is to test whether the learned
generators, especially absorbing-MASK diffusion, can exceed the simpler
single-pass Greedy reference through inference-time decoding improvements
without adding a hand-crafted search component.

## Current Probe Script

- Script: `scripts/92_probe_neural_decoding_enhancements.py`
- Policy: verified learned candidates are selected by exact latency; fallback
  is invoked only if no verified learned candidate exists.
- Included deterministic reference: `greedy` only.
- Excluded: latency-aware heuristic and local search.
- Supported learned probes:
  - Direct with larger proposal budgets and temperature scaling, e.g.,
    `direct_b128_t075`.
  - Sequential GNN with larger forward-equivalent budgets and temperature
    scaling, e.g., `sequential_b128_t075`.
  - Masked diffusion with larger K and temperature scaling, e.g.,
    `masked_k16_t1`, `masked_k32_t1`, `masked_k64_t1`.
  - Search-free decoding policies at the same nominal budget:
    - `mix`: one deterministic argmax proposal plus stochastic proposals.
    - `ens`: stochastic proposals split across temperatures 0.75, 1.0, and
      1.25.
    - `mixens`: one deterministic argmax proposal plus the temperature
      ensemble over the remaining proposal budget.

Use `t0p75` or `t075` for temperature 0.75.  The parser treats leading-zero
tokens such as `t075` as decimal temperatures.

## Local Smoke Results

These results are not paper evidence; they are only direction-finding probes on
`sealed_id`, seed `2026070113`, and 8 instances.

### Masked K Sweep at Temperature 1.0

- `greedy`: success 75.00%, Gap 1.34%.
- `masked_k8_t1`: success 100.00%, Gap 2.82%.
- `masked_k16_t1`: success 100.00%, Gap 1.55%.
- `masked_k32_t1`: success 100.00%, Gap 0.44%.
- `masked_k64_t1`: success 100.00%, Gap 0.35%.

Signal: increasing K can make masked diffusion exceed Greedy on this small
sample, but runtime rises from about 0.44 s at K=8 to about 1.76 s at K=32 and
3.49 s at K=64.

### Temperature/Budget Probe

- `direct_b128_t075`: success 100.00%, Gap 0.44%.
- `sequential_b128_t075`: success 100.00%, Gap 1.21%.
- `masked_k16_t075`: success 100.00%, Gap 2.43%.
- `masked_k16_t1`: success 100.00%, Gap 1.55%.
- `masked_k32_t075`: success 100.00%, Gap 1.72%.

Signal: lower temperature helped Direct and Sequential on this tiny sample but
did not help Masked Diffusion.  For Masked Diffusion, K appears more important
than lowering temperature.

### Same-Budget Decoding-Policy Probe

These are still not paper evidence.  They use `sealed_id`, seed `2026070113`,
and 32 instances.  The goal is to test whether reasonable decoding policies can
improve K=8 Masked Diffusion without adding search or a hand-crafted
constructive heuristic.

- `greedy`: success 87.50%, Gap 2.049%.
- `masked_k8_t1`: success 100.00%, Gap 2.442%, fallback 3.12%.
- `masked_k8_t1_mix`: success 100.00%, Gap 2.164%, fallback 3.12%.
- `masked_k8_t1_ens`: success 100.00%, Gap 2.157%, fallback 3.12%.
- `direct_b64_t1`: success 100.00%, Gap 3.270%, fallback 46.88%.
- `direct_b64_t1_ens`: success 100.00%, Gap 3.191%, fallback 53.12%.
- `sequential_b64_t1`: success 100.00%, Gap 2.413%, fallback 6.25%.
- `sequential_b64_t1_mix`: success 100.00%, Gap 2.260%, fallback 12.50%.

Signal: K=8 Masked Diffusion benefits from search-free decoding policies:
`mix` and `ens` reduce Gap from 2.442% to about 2.16% without changing K.
However, this local probe still does not clearly beat Greedy's 2.049% Gap,
although Greedy succeeds on only 87.50% of records.  Direct gains little from
the same-budget ensemble.  Sequential's `mix` variant improves Gap but increases
fallback use in this small sample.

Local controlled-shift testing was blocked because the local
`phase6e-e-controlled-shift` copy does not include `dataset_freeze.json`.
The complete controlled-shift/profiled evaluation should therefore be run on
the server or after resynchronizing the frozen dataset folders.

## Next Experiment Gate

Run a full server-side probe on the profiled workload before changing the
manuscript:

1. `greedy`
2. `direct_b64_t1`, `direct_b64_t1_ens`, `direct_b128_t075`
3. `sequential_b64_t1`, `sequential_b64_t1_mix`, `sequential_b128_t075`
4. `masked_k8_t1`, `masked_k8_t1_mix`, `masked_k8_t1_ens`,
   `masked_k16_t1`, `masked_k32_t1`

Primary decision criterion:

- If `masked_k16_t1` or `masked_k32_t1` consistently beats Greedy on profiled
  workload while preserving 100% final success, update the manuscript around a
  larger inference budget or budget-sensitivity result.
- If Direct/Sequential also beat Greedy at larger budgets, keep the main claim
  as a learned-generator comparison and emphasize Masked Diffusion only when it
  remains best among learned methods under the same forward-equivalent budget.
- If all learned methods still trail Greedy on profiled workload, keep Greedy as
  a strong deterministic reference and avoid claiming solver-level dominance.

## Remote Full Probe Started

Date: 2026-07-02

Remote host: `linchen@58.198.177.19`

Launcher:

- Local/remote script: `active_stage3/run_phase6f_decoding_policy_full.sh`
- Remote base output: `artifacts/phase6f-decoding-policy-full/`
- Status file: `artifacts/phase6f-decoding-policy-full/queue_status.tsv`
- Logs: `artifacts/phase6f-decoding-policy-full/logs/`

Settings launched:

- `sealed_id` on GPU 0, output `artifacts/phase6f-decoding-policy-full/sealed_id`
- `controlled_shift` on GPU 1, output `artifacts/phase6f-decoding-policy-full/controlled_shift`
- `realistic_profile` on GPU 2, output `artifacts/phase6f-decoding-policy-full/realistic_profile`

Methods:

- `greedy`
- `direct_b64_t1`, `direct_b64_t1_ens`
- `sequential_b64_t1`, `sequential_b64_t1_mix`
- `masked_k8_t1`, `masked_k8_t1_mix`, `masked_k8_t1_ens`
- `masked_k16_t1`

`masked_k32_t1` is intentionally excluded to keep online inference around the
sub-second target where possible.  The largest included masked setting is
`K=16`, i.e., \(B_{\mathrm{NN}}=128\).

## Remote Probe Status and Interim Results

Date: 2026-07-02

Completed remote settings:

- `sealed_id`: 40 records.
- `controlled_shift`: 200 records.
- `realistic_profile`: 120 records.

The completed reports were copied locally under
`artifacts/phase6f-decoding-policy-full/{sealed_id,controlled_shift,realistic_profile}/`.

Main signal from the completed settings:

- Masked Diffusion remains the strongest learned proposal generator in terms of
  verified-candidate availability.  Across completed settings, `masked_k8_t1`
  and `masked_k16_t1` usually produce feasible learned candidates for about
  96.50--100.00% of records, while Direct often requires fallback for about
  38.33--60.00% of records.
- `masked_k16_t1` improves the final Gap relative to `masked_k8_t1` on all
  completed settings, with online time around 0.87--1.04 s.
- The search-free `mix` and `ens` decoding variants are not uniformly helpful:
  `mix` helps on realistic profiles but not on sealed ID, and `ens` is slower
  without a consistent Gap improvement.  Do not promote them as a main method
  unless additional evidence changes this.
- Greedy remains a strong deterministic reference on Gap when it succeeds, but
  it has lower success than learned pipelines because it has no fallback in
  this probe.  This supports a conservative manuscript framing: emphasize
  Masked Diffusion's neural proposal feasibility and learned-baseline advantage,
  not solver-level dominance over every deterministic construction rule.

Cross-scale was added to
`scripts/92_probe_neural_decoding_enhancements.py` with dataset
`artifacts/datasets/phase6c-final-scale` and partitions
`scale_medium`, `scale_large`, and `scale_extra_large`.  A remote run was
started on GPU 3 with output
`artifacts/phase6f-decoding-policy-full/cross_scale`.

## Cross-Scale Probe Completed

Date collected: 2026-07-02

The cross-scale decoding-policy probe completed and was copied locally under
`artifacts/phase6f-decoding-policy-full/cross_scale/`.  The probe uses the
script default `--max-instances-per-partition=4`, giving 10 seeds, three scale
partitions, and 120 seed-instance records.  The local folder contains the
evidence JSON, report Markdown, all per-instance records, and
`cross_scale_result_summary_zh.md`.

Key cross-scale results:

- `masked_k16_t1`: success 100.00%, any feasible 94.17%, proposal feasible
  73.75%, Gap 5.09%, fallback 5.83%, time 3.223 s.
- `masked_k8_t1`: success 100.00%, any feasible 90.83%, proposal feasible
  74.06%, Gap 5.85%, fallback 9.17%, time 1.595 s.
- `masked_k8_t1_mix`: success 100.00%, any feasible 92.50%, proposal feasible
  74.38%, Gap 5.13%, fallback 7.50%, time 2.034 s.
- `direct_b64_t1`: success 100.00%, any feasible 6.67%, proposal feasible
  4.78%, Gap 7.40%, fallback 93.33%, time 0.862 s.
- `sequential_b64_t1`: success 100.00%, any feasible 49.17%, proposal feasible
  51.43%, Gap 9.11%, fallback 50.83%, time 1.335 s.
- `greedy`: success 58.33%, Gap 4.36%, time 0.003 s.

Interpretation: Masked Diffusion remains much stronger than Direct and
Sequential GNN at producing verified generated candidates under scale shift.
However, `masked_k16_t1` exceeds the informal one-second target on cross-scale.
For manuscript use, keep `K=8` / \(B_{\mathrm{NN}}=64\) as the conservative
main-budget setting unless higher cross-scale latency is acceptable, and report
`K=16` / \(B_{\mathrm{NN}}=128\) as a stronger-budget sensitivity result.
