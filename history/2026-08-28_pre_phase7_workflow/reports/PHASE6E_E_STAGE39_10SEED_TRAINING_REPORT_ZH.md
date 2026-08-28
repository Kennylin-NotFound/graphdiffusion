# Phase 6E-E Stage 3.9 十种子训练汇总

更新时间：2026-06-24

## 状态

- 远程训练队列：已完成。
- 远程路径：`~/GDM_Paper/implementation/artifacts/phase6e-e-stage39-10seed-training`
- 新增云端训练种子：`2026070114`--`2026070120`
- 合并的既有 Stage 3.8 冻结种子：`2026070111`--`2026070113`
- 训练失败数：`0`
- 当前证据层级：checkpoint-selection / training evidence。
- 论文主结果仍需单独执行十种子的统一 final evaluation / freeze。

## 十种子训练选择指标

指标含义：

- `Raw`：至少一个原始神经 proposal 可行的实例比例。
- `Pre-gap`：fallback 之前的平均 pre-fallback latency gap，越低越好。
- `Pre-succ`：fallback 之前能够得到候选解的实例比例。
- `Time`：checkpoint-selection 阶段记录的平均在线时间；该值受硬件与实现影响，不能直接作为跨硬件公平结论。

| Seed | Direct Raw (%) | Masked Raw (%) | Raw Gain (pp) | Direct Pre-gap (%) | Masked Pre-gap (%) | Rel. Gap Improve (%) | Direct Pre-succ (%) | Masked Pre-succ (%) | Direct Time (s) | Masked Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026070111 | 51.56 | 92.19 | 40.62 | 6.837 | 3.231 | 52.75 | 98.44 | 98.44 | 0.0165 | 0.1332 |
| 2026070112 | 46.88 | 98.44 | 51.56 | 5.611 | 4.682 | 16.56 | 98.44 | 100.00 | 0.0186 | 0.3538 |
| 2026070113 | 42.19 | 95.31 | 53.12 | 6.245 | 3.932 | 37.04 | 100.00 | 100.00 | 0.0454 | 0.3509 |
| 2026070114 | 39.06 | 95.31 | 56.25 | 7.058 | 1.735 | 75.42 | 98.44 | 96.88 | 0.0381 | 0.1648 |
| 2026070115 | 40.62 | 95.31 | 54.69 | 5.506 | 2.303 | 58.18 | 98.44 | 98.44 | 0.0260 | 0.2206 |
| 2026070116 | 40.62 | 95.31 | 54.69 | 7.271 | 2.149 | 70.44 | 96.88 | 98.44 | 0.0392 | 0.2415 |
| 2026070117 | 46.88 | 95.31 | 48.44 | 6.565 | 1.376 | 79.04 | 98.44 | 98.44 | 0.0347 | 0.1695 |
| 2026070118 | 43.75 | 93.75 | 50.00 | 6.105 | 5.357 | 12.25 | 98.44 | 100.00 | 0.0353 | 0.2417 |
| 2026070119 | 42.19 | 95.31 | 53.12 | 7.092 | 1.841 | 74.04 | 98.44 | 98.44 | 0.0358 | 0.2188 |
| 2026070120 | 45.31 | 96.88 | 51.56 | 6.487 | 2.047 | 68.45 | 98.44 | 98.44 | 0.0377 | 0.2211 |

## 聚合结果

| Metric | Direct Mean | Masked Mean | Difference / Note |
|---|---:|---:|---:|
| Raw feasibility (%) | 43.91 | 95.31 | +51.41 pp |
| Pre-fallback gap (%) | 6.478 | 2.865 | 55.77% lower in absolute mean comparison |
| Mean relative gap improvement (%) | - | - | 54.42 |
| Pre-fallback success (%) | 98.44 | 98.75 | +0.31 pp |
| Online time (s) | 0.0327 | 0.2316 | masked is slower in wall-clock time |

Seed-level checks:

- Masked raw feasibility is higher on `10/10` seeds.
- Masked pre-fallback gap is lower on `10/10` seeds.
- Masked pre-fallback success is not lower on `9/10` seeds.

## 当前可支撑的表述边界

可以支撑：

- 在 checkpoint-selection 层面，absorbing-MASK conditional diffusion 明显产生更高比例的原始可行 proposal。
- 在同一训练选择评估上，masked diffusion 的 pre-fallback latency gap 也更低，且该趋势在十个种子上保持一致。
- 这说明新的 masked diffusion 设计相比 one-shot direct predictor 更适合生成结构化可行部署候选。

暂不能直接支撑：

- 不能把这份训练选择指标直接写成最终论文主结果。
- 不能仅凭 wall-clock time 指标主张跨硬件效率优势。
- 还需要运行统一的 final evaluation / freeze，尤其是 same neural forward-equivalent budget 下的 sealed evaluation，并统计 paired tests。

## 下一步

1. 运行十种子的统一 final evaluation / freeze。
2. 输出 raw-only、pre-fallback、post-repair/fallback、final verified gap 四层指标。
3. 对 same neural forward-equivalent budget 结果执行 paired sign test 或等价的配对统计检验。
4. 再决定论文中是否采用“同等神经前向预算下，masked diffusion 生成更高质量可行候选”的核心叙事。
