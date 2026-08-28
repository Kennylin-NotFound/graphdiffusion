# Phase 6E-E Stage 3.6 随机解码效率校准报告

更新时间：2026-06-22

## 1. 任务与证据边界

Stage 3.6A 只使用已冻结的 64 个 `checkpoint_selection` 实例，未重新访问已消费的 `pilot_gate`，也未生成 Stage 3 final data 或新的 confirmation data。模型 checkpoint、八步 conditional schedule、temperature、hard mask、repair、fallback、精确目标函数和 Gate R3-C 阈值均保持不变。

本阶段测试 deterministic `K=1` 与 stochastic `K=2/4/8`。选择规则预先固定为：只有同时满足至少 5% pre-fallback gap 改善、配对胜多于负、raw feasibility 与最终成功率不下降，以及总时延比不超过 1.10，候选 K 才可通过；若多个 K 通过，则选择最小 K。

## 2. 校准结果

| Method | Final gap | Pre-fallback gap | Raw feasible | Total time | Time ratio |
|---|---:|---:|---:|---:|---:|
| Deterministic K=1 | 1.257% | 3.231% | 92.19% | 0.291 s | 1.000 |
| Stochastic K=2 | 1.332% | 3.575% | 93.75% | 0.323 s | 1.108 |
| Stochastic K=4 | 0.995% | 2.351% | 96.88% | 0.373 s | 1.280 |
| Stochastic K=8 | 0.871% | 1.646% | 96.88% | 0.469 s | 1.610 |

K=2 的 pre-fallback gap 相对恶化 10.65%，配对结果为 17 胜、21 负、26 平，并且时延比 1.108 仍略高于上限。K=4 改善 27.25%，配对结果为 26 胜、12 负、26 平，但时延比为 1.280。K=8 改善 49.06%，配对结果为 30 胜、6 负、28 平，但时延比为 1.610。

因此，没有候选同时通过质量和近等时延条件。冻结选择为 `selected_method=null`，`confirmation_authorized=false`。

## 3. 科学解释

结果呈现稳定的 proposal-budget tradeoff：K 增大后，随机解码的候选质量、配对胜率和 raw feasibility 整体改善，但计算开销同步增加。K=2 还不足以稳定利用随机性，K=4/K=8 则不能满足预注册的近等时延要求。

这不是“随机解码完全无效”。它说明随机多候选搜索可以换取质量，但现有实现没有证明该收益能以接近 deterministic completion 的成本取得。因此论文不能把扩散采样写成近等成本下优于 sequential conditional baseline 的独立优势。

## 4. 冻结结论与下一步

Stage 3.6 的 diffusion-specific efficiency branch 在此关闭，不生成预注册的 128-instance confirmation 数据，也不修改阈值继续搜索。Stage 3 的最强可辩护主线为 Outcome B：typed factor graph 上的 deterministic partial-assignment conditional generator 相比重训练 direct predictor 获得明确改善；stochastic decoding 只能作为可选的质量与时延权衡。

下一阶段应冻结 Outcome-B production contract，规划多种子 deterministic masked/direct 训练与新的 sealed evaluation，而不是继续在已使用的开发数据上调 stochastic K。

冻结证据：

- `artifacts/phase6e-e-stage36-calibration/efficiency_evidence.json`
- evidence SHA-256: `A12FED3474A212264F6DBAB6D4794F8E8CE20BDD0A7DC057DE0F70CDE055F3D8`
- `artifacts/phase6e-e-stage36/efficiency_selection_lock.json`
- selection SHA-256: `48256B875A333259C13D448D9CBC8B84A9746D4DC306BAEDA8BD90710D7F2719`

