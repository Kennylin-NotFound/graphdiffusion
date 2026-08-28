# Phase 6E-E Stage 3 一次性 Pilot 验收报告

更新时间：2026-06-22

## 1. 验收结论

Stage 3 一次性 `pilot_gate` 已按冻结合约完成。评估使用 64 个此前未访问的实例、固定检查点、固定推理预算，以及相同的验证、修复、fallback 和精确目标计算。完整回归为 `121 passed`。证据冻结在
`artifacts/phase6e-e-stage3-pilot/pilot_evidence.json`，SHA-256 为
`766DB12C1BE0DE5A37EFBA78FA195FB854B0531D480058F603435C930239E8CB`。

预注册结果为 **Outcome B**：确定性 partial-assignment conditioning 通过 Gate R3-B；随机 masked decoding 的质量指标通过 Gate R3-C，但在线时延比超过上限，因此 Gate R3-C 整体未通过。

## 2. Pilot 结果

| Method | Final gap | Pre-fallback gap | Raw feasible | Total time |
|---|---:|---:|---:|---:|
| Retrained direct, K=32 | 1.467% | 4.660% | 48.44% | 0.228 s |
| Deterministic masked, K=1 | 1.439% | 3.226% | 95.31% | 0.294 s |
| Stochastic masked, K=8 | 0.926% | 2.221% | 98.44% | 0.477 s |
| Historical diffusion, K=4 | 0.987% | 1.751% | 54.69% | 0.982 s |
| Historical trajectory rescue | 0.650% | 0.882% | 65.63% | 0.997 s |

所有方法在完整 hybrid pipeline 下均达到 100% 最终成功率。由于合约始终生成 fallback 候选，该指标不能用于证明神经生成器本身的可行性；应结合 raw feasibility、pre-fallback gap、最终 gap 和时延解释。

## 3. Gate 解释

### Gate R3-B：Partial Conditioning

- pre-fallback gap 相对改善：30.77%；
- raw feasibility 提高：46.875 个百分点；
- 总时延比：1.289，不超过 1.50；
- 最终 verified success 未下降。

Gate R3-B 全部通过。这说明显式 partial assignment、剩余容量和已提交依赖上下文，相比相同数据与训练预算下的一次性 direct predictor，确实改善了部署决策。最终 gap 的差异较小，是因为共同 fallback 会压缩两者在最终选择阶段的差距；因果解释应以 pre-fallback 指标为主。

### Gate R3-C：Diffusion-Specific Value

- pre-fallback gap 相对改善：31.14%；
- 配对结果：40 胜、8 负、16 平；
- raw feasibility 未下降；
- 总时延比：1.620，高于 1.10。

随机 masked decoding 明显提高候选质量，但未在预注册的近等时延条件下完成该提升。因此现有结果支持“随机多候选解码提供质量与时延权衡”，不支持“扩散过程以近等成本优于确定性 conditional completion”。

## 4. 历史方法边界

新的 stochastic masked 方法在本 pilot 上比旧 categorical diffusion 更快且最终 gap 略低，但两者的训练数据、状态定义和模型家族不同。该比较只能作为历史上下文，不能替代 retrained direct 与 deterministic masked 两个因果控制组。Trajectory rescue 仍获得最低 gap，但耗时约为新 stochastic masked 的两倍。

## 5. 下一阶段

不得重新使用本 pilot 调参，也不得修改 Gate 阈值追逐正结果。下一步应先在 `checkpoint_selection` 上分析 stochastic K=2/K=4、批处理和重复图构建的时延组成，冻结一个更接近确定性时延的随机解码预算，再使用全新种子的独立 confirmation contract 做一次验证。只有该效率门槛通过，才进入三种子训练和最终 sealed evaluation；否则论文应以 typed graph partial-assignment conditional generator 为核心，并将随机解码作为可选的质量预算。
