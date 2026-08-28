# Phase 6E-E Stage 3.9 十种子 Forward-Budget 评估报告

更新时间：2026-06-24

## 结论摘要

Stage 3.9 已完成十个训练 seed 的统一 sealed evaluation。主公平口径已从
time-matched 切换为 same neural forward-equivalent budget：

\[
B_{\mathrm{NN}} = N_{\mathrm{prop}} \times N_{\mathrm{step}} .
\]

其中 Direct K=64 使用一次式预测，\(B_{\mathrm{NN}}=64\)；Masked Diffusion
K=8 使用 8 个 absorbing-MASK completion steps，\(B_{\mathrm{NN}}=64\)。

在该硬成本口径下，Masked Diffusion 的 raw proposal 质量和 pre-fallback 质量
已经形成稳定、统计显著的优势；full pipeline 的平均 final gap 也更低，但 final
paired sign test 尚未达到 0.05。因此论文可以更强地表述为：

> 在相同神经前向等价预算下，absorbing-MASK graph diffusion 比一次式 Direct
> predictor 生成更高可行性、更低 pre-fallback latency gap 的部署候选；经过相同
> hard verification、bounded repair 和 fallback 后，完整 pipeline 也呈现更低平均
> final gap，但最终端到端统计显著优势仍需谨慎表述。

## 证据位置

- 远程评估已完成：`1280/1280` seed-instance records。
- 本地 evidence：
  `implementation/artifacts/phase6e-e-stage39-forward-budget-evaluation/forward_budget_evidence.json`
- 本地英文报告：
  `implementation/artifacts/phase6e-e-stage39-forward-budget-evaluation/forward_budget_report.md`
- 本地完整 records：
  `implementation/artifacts/phase6e-e-stage39-forward-budget-evaluation/records/`
- 十种子 training freeze：
  `implementation/artifacts/phase6e-e-stage39-10seed-training/ten_seed_training_freeze.json`
- 评估脚本：
  `implementation/scripts/74_run_phase6e_e_stage39_forward_budget.py`

## 主比较：Direct K=64 vs Masked Diffusion K=8

| Method | \(B_{\mathrm{NN}}\) | Raw success | Raw gap | Repair-only gap | Full gap | Source(raw/repair/fallback) | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct K=64 | 64 | 49.69% | 3.585% | 3.357% | 1.431% | 7.50% / 29.06% / 63.44% | 1.088 s |
| Masked Diffusion K=8 | 64 | 96.80% | 2.557% | 2.524% | 1.232% | 37.50% / 1.64% / 60.86% | 0.661 s |

配对检验按 sealed instance 聚合，并对十个训练 seed 取平均后比较：

| Stage | Masked wins | Direct wins | Ties | p-value | 解释 |
|---|---:|---:|---:|---:|---|
| Raw proposal | 120 | 7 | 1 | \(1.11\times 10^{-27}\) | 极强支持 masked raw proposal 优势 |
| Pre-fallback | 76 | 49 | 3 | 0.0197 | 支持 masked 在 fallback 前质量更好 |
| Final full pipeline | 46 | 32 | 50 | 0.1405 | 平均 gap 更低，但不宜写统计显著端到端优势 |

## Baseline Suite

| Method | \(B_{\mathrm{NN}}\) | Success | Final gap | Raw success | Fallback selected | Time |
|---|---:|---:|---:|---:|---:|---:|
| Fallback only | 0 | 100.00% | 3.364% | 0.00% | 100.00% | 0.009 s |
| Random K=64 | 0 | 100.00% | 2.483% | 41.88% | 81.41% | 0.983 s |
| Direct K=64 | 64 | 100.00% | 1.431% | 49.69% | 63.44% | 1.088 s |
| Masked deterministic K=1 | 8 | 100.00% | 1.546% | 95.16% | 65.31% | 0.462 s |
| Masked Diffusion K=8 | 64 | 100.00% | 1.232% | 96.80% | 60.86% | 0.661 s |

这个 baseline suite 补足了 checklist 中的 P1-A：fallback-only / constructive
heuristic、random compatible proposals、Direct、Masked deterministic 和 Masked
Diffusion 均在同一 sealed set 上完成。

## Post-Processing Ablation

本次同一协议同时输出 raw-only、repair-only 和 full pipeline 三层结果，补足
P1-B。主锚点显示：

- Direct 从 raw 到 repair-only 的 gap 为 3.585% -> 3.357%，说明 repair 有帮助，
  但最终仍高度依赖 fallback，fallback selected 为 63.44%。
- Masked Diffusion 从 raw 到 repair-only 的 gap 为 2.557% -> 2.524%，repair
  负担很小，raw source 选择比例达到 37.50%，repair source 仅 1.64%。
- 两者最终 success 都是 100%，但该 success 是 shared hard pipeline 的结果，
  不能解释为 neural model 单独保证 feasibility。

## Forward-Budget Sensitivity

| \(B_{\mathrm{NN}}\) | Direct full gap | Masked full gap | Direct raw success | Masked raw success |
|---:|---:|---:|---:|---:|
| 8 | 1.748% | 1.546% | 43.05% | 95.16% |
| 16 | 1.636% | 1.614% | 45.08% | 95.94% |
| 32 | 1.545% | 1.376% | 47.58% | 96.48% |
| 64 | 1.431% | 1.232% | 49.69% | 96.80% |
| 128 | 1.332% | 1.067% | 51.80% | 97.03% |

该结果补足 P1-C，并且比 wall-clock time-matched 更适合作为核心模型比较口径：
在相同 \(B_{\mathrm{NN}}\) 下，Masked Diffusion 在所有预算点上都取得更低 full
gap 和更高 raw success。

## 写作边界

可以写：

- Same neural forward-equivalent budget 下，Masked Diffusion 生成的 raw
  deployment proposals 显著更可行、质量更高。
- Pre-fallback 层面 Masked Diffusion 已通过配对统计检验。
- Full pipeline 平均 final gap 更低，并且在多个 \(B_{\mathrm{NN}}\) 点上保持
  一致趋势。
- Feasibility 是 neural proposal、hard verifier、bounded repair 和 deterministic
  fallback 共同保证的 pipeline property。

仍需谨慎：

- 不写 statistically confirmed final end-to-end superiority，除非后续 final
  paired test 或额外实验也达到显著。
- 不写真实部署或真实 trace 验证。
- 不写 learned solver 超过 MILP；MILP 仍是 solution-pool reference。

## 后续建议

1. 生成论文用表：主表建议使用 \(B_{\mathrm{NN}}=64\) 的 raw / repair-only /
   full 分层结果。
2. 生成 sensitivity 图或表：展示 \(B_{\mathrm{NN}}=8,16,32,64,128\)。
3. 可选补做 controlled-shift / realistic-profile 小实验，用于支撑 robustness 或
   practical scenario 叙述。
4. 暂不再强调旧 energy/guidance 消融，除非正文重新把它们定义为核心贡献。
