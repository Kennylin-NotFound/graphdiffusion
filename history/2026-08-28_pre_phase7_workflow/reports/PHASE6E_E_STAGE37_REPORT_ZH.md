# Phase 6E-E Stage 3.7 Diffusion 近等时延开发门槛报告

更新时间：2026-06-22

## 1. 核心问题

本阶段检验的方法级命题是：在相近的总在线推理预算下，absorbing-MASK graph diffusion 的结构化 reverse completion 是否能够比增加 one-pass Direct proposals 获得更高质量的部署方案。

该命题不同于 Stage 3.6 的 stochastic-versus-deterministic 效率门槛。Deterministic masked K=1 在这里是部分状态条件机制的消融；时间匹配的 Direct predictor 才是主 baseline。

## 2. 有界优化预检

新增向量化 decoder 在 8 个预检实例上将平均 K=8 sampling time 降低至 legacy decoder 的 68.92%，且 deterministic 输出完全一致。但一个实例中只有 7/8 个随机 proposal 完成。由于预注册规则要求每个向量化 proposal 都必须完成，该优化路径未被接受，正式 Stage 3.7 使用 legacy decoder。

这意味着后续正结果不依赖尚未完全验收的工程优化。

## 3. 64-Instance Checkpoint-Selection 结果

Direct proposal count 仅按平均总时延与 Diffusion K=8 的距离选择。K=64 最接近，因此成为冻结的 matched-time baseline。

| Method | Pre-fallback gap | Final gap | Raw feasible | Total time |
|---|---:|---:|---:|---:|
| Direct K=64 | 4.493% | 0.930% | 59.38% | 0.462 s |
| Deterministic masked K=1 | 3.231% | 1.257% | 92.19% | 0.315 s |
| Masked Diffusion K=8 | 1.647% | 0.806% | 96.88% | 0.506 s |

Diffusion/Direct 总时延比为 1.095，处于预注册的 0.90--1.10 区间。Diffusion 的 pre-fallback gap 相对改善 63.35%，配对结果为 42 胜、10 负、12 平；final gap、raw feasibility 和 final success 均不差于 Direct。全部 matched-time checks 通过。

## 4. 当前可支持的叙事

开发证据支持以下机制链：Direct predictor 通过增加独立完整部署样本扩展搜索，而 absorbing-MASK diffusion 在 reverse completion 中让后续服务决策显式依赖已提交部署、剩余资源和可见依赖关系。即使 Direct 使用更多 proposals 并匹配总在线时间，结构化 reverse completion 仍产生更高质量的候选。

如果新的 sealed multi-seed evaluation 复现该结果，论文可以主张：

> The proposed graph diffusion solver achieves higher deployment quality than direct prediction under a comparable total online budget.

该表述是整体 diffusion-based solver 的优势，不等同于“stochastic sampling 比 deterministic completion 更高效”。后者仍应报告为质量与时延权衡。

## 5. 决策与下一步

Stage 3.7 development gate 通过，`sealed_multiseed_authorized=true`。下一阶段可以预注册新的 sealed 数据、至少三个训练种子、Direct K=64、deterministic masked K=1 和 Masked Diffusion K=8，并在不再调整方法的前提下完成正式评估。

现有 pilot、checkpoint-selection 和历史 final partitions 均不得继续用于调参。当前结果在 sealed multi-seed 复现前不能写成最终实验结论。

冻结证据：

- `artifacts/phase6e-e-stage37-calibration/matched_time_evidence.json`
- evidence SHA-256: `6B619CDD565BF503906E688E90BAAB20F78529C9493A4364B1C733855197450D`
- `artifacts/phase6e-e-stage37/matched_time_decision_lock.json`
- decision SHA-256: `D844E7E0622A8142A70D7EC0467DD4197F49A8B7EBD6757D35AB5047BC513D39`

