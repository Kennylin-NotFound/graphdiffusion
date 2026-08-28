# Phase 6E-E Stage 3 单种子训练验收报告

更新时间：2026-06-22

## 1. 验收结论

direct control 与 masked conditional 两个正式训练任务均完成 20,000 步，
训练配置、数据集、seed、batch size、目标策略和优化步数一致。两个 run 均生成
`best.pt`、`latest.pt`、完整 `metrics.jsonl` 和配置快照，无 NaN、异常退出或缺失
checkpoint。训练后完整回归为 117 passed。

训练成果已冻结至
`artifacts/phase6e-e-stage3-training/training_freeze.json`，SHA-256 为
`9FA9FB25C24977523323DAAE0740E5B7BECEC39027E546605954B70C6A9CD4EB`。
64-instance pilot gate 尚未开启，因此本报告不是 Stage 3.5 最终方法验收。

## 2. 最优 Checkpoint

| Model | Best step | Pre-fallback success | Pre-fallback gap | Raw feasibility | Online time |
|---|---:|---:|---:|---:|---:|
| Direct | 8,000 | 98.44% | 6.837% | 51.56% | 0.0165 s |
| Masked conditional | 3,000 | 98.44% | 3.231% | 92.19% | 0.1332 s |

以上数值仅来自与训练集分离的 `checkpoint_selection` split。相对 direct，
masked conditional 的 pre-fallback gap 降低 52.75%，raw feasibility 提高
40.62 个百分点，但在线时间为 direct 的 8.06 倍。两者最终 verified rate
均为 100%，不能用该指标区分神经生成器，因为 fallback 会掩盖原始失败。

## 3. 训练行为

- direct 最优 checkpoint 出现在 8,000 步，masked conditional 出现在 3,000 步；
  checkpoint selection 成功避免使用过拟合更严重的最终权重。
- 20,000 步时两个训练损失均接近零，说明模型充分拟合训练集，也提示后续结论
  必须依赖未参与 checkpoint 选择的 pilot split。
- 实际运行时间约为 direct 33.9 分钟、masked conditional 38.6 分钟。
- 两个配置快照字节一致，均使用 seed `2026070111`、best-only targets、batch
  size 16 和相同的 20,000-step budget。

## 4. Gate 边界与下一步

checkpoint-selection 结果支持“partial conditioning 可能改善部署相关性”的
假设，但不构成 Gate R3-B 通过证据。当前 masked/direct 在线时间比 8.06 已明显
高于预注册的 1.5 上限，因此在开启一次性 pilot 前，必须仅在
`checkpoint_selection` 上完成推理时延口径复核和预注册预算校准。不得修改已冻结
checkpoint、训练目标或 Gate 阈值来追逐正结果。

下一步应实现可复现的 Stage 3 pilot/freeze runner，先锁定 direct one-shot、
deterministic conditional、stochastic masked 和历史 anchor 的推理合同，再一次性读取
`pilot_gate`。只有 pilot 结果才能决定是否进入多种子训练。

