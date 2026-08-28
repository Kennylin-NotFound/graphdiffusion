# Phase 6E-E Stage 3.8 Layered Comparison

Updated: 2026-06-23

Source:
`implementation/artifacts/phase6e-e-stage38-sealed-evaluation/final_evidence.json`
and the 384 per-instance records under
`implementation/artifacts/phase6e-e-stage38-sealed-evaluation/records/`.

## 分层主表

| Method | Raw-any feasible | Mean raw-feasible proposal rate | Pre-fallback success | Pre-fallback gap | Final success | Final gap | Total time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct K=64 | 48.96% | 34.65% | 98.18% | 3.693% | 100.00% | 1.631% | 0.501 s |
| Masked Det. K=1 | 94.53% | 94.53% | 97.92% | 4.020% | 100.00% | 1.589% | 0.305 s |
| Masked Diff. K=8 | 95.57% | 94.27% | 98.18% | 2.912% | 100.00% | 1.254% | 0.486 s |

## Raw-only 补充评估

该补充评估复用 Stage 3.8 的冻结模型、sealed dataset、采样预算和原始 sampling seed
namespace，但关闭 bounded repair 与 deterministic fallback。因此 `Raw gap` 是
raw proposals 中最佳可行候选相对 MILP solution-pool best 的 gap。

| Method | Raw success | Raw feasible proposal rate | Raw gap | Capacity violation | Link violation | Total time |
|---|---:|---:|---:|---:|---:|---:|
| Direct K=64 | 48.96% | 34.65% | 4.009% | 44.50% | 41.42% | 0.303 s |
| Masked Det. K=1 | 94.53% | 94.53% | 4.042% | 0.00% | 0.00% | 0.295 s |
| Masked Diff. K=8 | 95.57% | 94.27% | 2.985% | 0.00% | 0.00% | 0.482 s |

Direct K=64 与 Masked Diff. K=8 的 raw-only paired comparison 为 110 wins、
12 losses、6 ties，sign-test p-value 为 `5.47e-21`。Masked Diff. K=8 的
raw gap 相对 Direct K=64 改善 25.54%，raw success rate 提升 46.61 个百分点。

## 最终选中候选来源

| Method | Selected raw | Selected repair | Selected fallback |
|---|---:|---:|---:|
| Direct K=64 | 7.03% (27/384) | 27.34% (105/384) | 65.62% (252/384) |
| Masked Det. K=1 | 31.25% (120/384) | 1.56% (6/384) | 67.19% (258/384) |
| Masked Diff. K=8 | 36.72% (141/384) | 2.08% (8/384) | 61.20% (235/384) |

## Seed-level 分解

| Seed | Method | Raw-any feasible | Pre-fallback gap | Final gap | Total time |
|---|---|---:|---:|---:|---:|
| 2026070111 | Direct K=64 | 46.88% | 4.153% | 1.733% | 0.508 s |
| 2026070111 | Masked Det. K=1 | 93.75% | 3.817% | 1.647% | 0.304 s |
| 2026070111 | Masked Diff. K=8 | 96.88% | 2.142% | 1.163% | 0.486 s |
| 2026070112 | Direct K=64 | 50.78% | 3.445% | 1.486% | 0.513 s |
| 2026070112 | Masked Det. K=1 | 93.75% | 4.858% | 1.805% | 0.305 s |
| 2026070112 | Masked Diff. K=8 | 93.75% | 4.133% | 1.540% | 0.487 s |
| 2026070113 | Direct K=64 | 49.22% | 3.474% | 1.673% | 0.481 s |
| 2026070113 | Masked Det. K=1 | 96.09% | 3.401% | 1.315% | 0.304 s |
| 2026070113 | Masked Diff. K=8 | 96.09% | 2.483% | 1.058% | 0.484 s |

## 解释边界

- Raw-any feasible 与 mean raw-feasible proposal rate 只衡量神经生成器的原始候选，
  不包含 repair 与 fallback。Raw-only 补充评估进一步确认 Masked Diff. K=8
  在这一层明显优于 Direct K=64。
- Pre-fallback 指 raw proposals 加 bounded repair 后、fallback 之前的最佳候选。
  它仍然不包含 deterministic fallback。
- Final gap 是完整 solver pipeline 的结果，包含 raw、repair、fallback 和 exact
  latency selection。因此 final 层优势会被后处理缩小。
- Direct K=64 的 selected repair 比例为 27.34%，而 Masked Diff. K=8 只有
  2.08%。这说明 repair 主要在补 Direct raw feasibility 的短板。
- Stage 3.8 仍然不能支持无条件的 diffusion-superiority claim，因为严格 gate 的
  sign test 未通过，且 seed 2026070112 在 pre-fallback/final gap 上反向。

## 论文可用表述

可以将结论写成：Masked Diffusion substantially improves the feasibility of
raw neural proposals and reduces the reliance on repair. After the shared
repair/fallback layer is enabled for all methods, the final verified latency
advantage becomes smaller but remains favorable in aggregate.

不应写成：Masked Diffusion is statistically confirmed to dominate Direct K=64
as an end-to-end solver under the locked comparable-time gate.
