# Phase 6E-E Stage 3 训练前验收报告

更新时间：2026-06-21

## 1. 结论

Stage 3 已完成正式训练之前的全部工程步骤。新的 absorbing-MASK
partial-assignment 模型族、确定性/随机条件解码器、独立 checkpoint 契约、
训练入口和开发数据均已实现并冻结。尚未启动 20,000-step 正式训练，也未读取
或生成新的 final-ID 数据。

## 2. 实现边界

- 保留原 typed factor graph、精确时延计算、硬验证器、repair 和 fallback。
- 新增显式 `PartialPlacementState`，区分未提交 MASK 与 batch padding。
- 前向过程采用端点严格的 absorbing-MASK schedule；masked CE 仅作用于隐藏服务。
- partial context 包含已提交处理时延、设备负载/剩余容量以及可见依赖的链路信息。
- deterministic conditional 与 stochastic masked decoding 使用相同模型、相同
  unmask schedule、相同硬容量/链路掩码和相同后处理。
- 正式比较必须同时包含使用相同数据、目标策略、图主干维度和训练步数的
  retrained direct control。

## 3. 工程验收

- 新增 Stage 3/归档兼容性定向测试 18 项；最终全项目回归为 117 项。
- CUDA toy overfit 500 步：loss 从 `0.761256` 降至 `0.000041`。
- 单隐藏、部分隐藏和全隐藏状态的最差 hidden-service accuracy 为 `100%`。
- checkpoint 恢复后的 logits 逐元素一致。
- deterministic 与 stochastic decoding 均通过硬验证和精确目标选择。
- 正式训练入口对 masked/direct 两种模型完成无 optimizer step 的 CUDA preflight。
- 参数量：masked conditional `920,833`，direct control `901,505`，差异约
  `2.14%`；后续报告需同时给出参数量和在线时间。

## 4. 开发数据

- 数据集：`phase6e-e-stage3-development`。
- split：train 256、checkpoint-selection 64、untouched pilot-gate 64。
- 三个 split 均为相同的 in-distribution 生成范围，实例 ID 和生成 seed 无重叠。
- 共 384 个实例、6106 个逐项验证的 MILP 部署标签。
- 381 个实例达到目标池深度 16；3 个实例在穷尽全部可行部署后分别包含
  3、1、6 个解。Stage 3 使用 best-only target，因此监督目标定义不受影响。
- dataset manifest、solution-pool manifest、精确目标与硬可行性审计均通过，
  `dataset_freeze.json` 已生成。

## 5. 训练与选择契约

- 训练 seed：`2026070111`。
- masked/direct 使用相同 batch stream、best-only target、hidden dimension、
  message-passing depth、optimizer、batch size 和 20,000-step budget。
- checkpoint 仅使用 `checkpoint_selection` split；排序依次考虑 pre-fallback
  success、pre-fallback gap、raw feasibility 和 online time，避免 fallback 掩盖
  原始模型失败。
- `pilot_gate` 不参与模型、checkpoint、温度或预算选择，只能在两种模型训练和
  checkpoint 冻结后开启一次。
- 未来 Stage 3 final contract 的 seed 已预注册，但数据尚未生成。

## 6. 下一步

按顺序训练 direct control 和 masked conditional model，冻结两个最佳 checkpoint
及训练曲线后，再编写并执行一次性 pilot-gate 对比。Stage 3.5 通过前不修改论文
Section IV/V，也不解封任何 final 数据。

## 7. 活动入口与归档

- 唯一正式训练入口：`active_stage3/run_training.ps1`；它会先验证训练前 freeze，
  再显式调用新 Stage 3 配置和 Python 入口。
- 旧阶段脚本与配置已移至 `history/pre_stage3_2026-06-21/`，仅用于历史证据复验。
- 活动目录现保留 12 个 Stage 3/通用脚本（含训练后 finalizer）和 3 个锁定配置。
- 117 项测试全部保留，最终回归耗时 7.87 秒。
- 内部 Python trainer 会拒绝未经过活动 launcher 授权的直接调用。
- 当前 training-ready freeze SHA-256 为
  `955E29AD07D1C91A9D6AEC0895BE5F001737C5735F3050A6C58791855B0B666B`。
