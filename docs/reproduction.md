# GDM 复现范围与入口

## 无历史数据的最小运行

按根目录 README 安装依赖，执行数据生成、数据审计和四组基础测试。小数据集可进一步通过 Gurobi 标注（需自行配置许可证）：

```bash
python scripts/26_label_contract_dataset.py artifacts/public_smoke
```

小样例不是论文训练集，不能直接替换冻结评估集后继续引用论文数值。

## 完整 Phase 7 评估的前提

以下资产目前仅保存在作者本地，未发布下载包。必须先取得相互匹配的数据、权重和哈希清单，或按公开源码重新生成、训练并建立新的实验批次；重新训练不保证逐项重现历史数值。

| 所需路径 | 用途 |
|---|---|
| `artifacts/datasets/phase6e-e-stage38-sealed/` | 主评估数据、标签池及冻结清单 |
| `artifacts/datasets/phase6e-e-controlled-shift/` | 受控分布变化 |
| `artifacts/datasets/phase6c-final-scale/` | 跨规模评估 |
| `artifacts/datasets/phase6e-e-realistic-profile/` | 现实特征仿真配置数据 |
| `artifacts/phase6e-e-stage39-10seed-training/ten_seed_training_freeze.json` | Masked / Direct 十种子冻结索引及其引用的权重 |
| `artifacts/phase6f-sequential-conditional-training/` | Sequential GNN 权重与清单 |
| `artifacts/phase6e-e-controlled-shift-evaluation-10seed/records/2026070111/` | 受控变化的 MILP 参考记录 |
| `artifacts/phase6e-e-realistic-profile-evaluation-10seed/records/2026070111/` | 现实特征仿真的 MILP 参考记录 |

冻结索引引用的文件也必须齐备；不能只放置索引 JSON。实际依赖以 Script 97 的 `DATASET_SPECS`、Script 92 的采样实现及冻结清单为准。

已有匹配资产时，可先执行单设置、单种子的有界检查：

```bash
python scripts/97_run_phase7_proposal_recovery_evaluation.py run --settings main_evaluation --seeds 2026070111 --device cpu --max-instances-per-partition 1 --no-skip-missing-datasets --output-root artifacts/reproduction_check
```

该命令仍需要对应权重和主评估数据。完整评估应去掉实例限制，使用论文所需全部设置和种子，并在相同硬件、预算及精度条件下比较效率；不要把 CPU 与原 GPU 时延直接比较。`--no-skip-missing-datasets` 用于让缺失数据明确报错。

## 从头训练的代码路径

1. `03_generate_dataset.py`：按 YAML 生成实例；`26_label_contract_dataset.py`：按数据合同构造 MILP 解池。
2. `18_freeze_labeled_dataset.py`：生成训练使用的冻结信息。
3. `64_train_phase6e_e_stage3.py`：Masked / Direct 训练；`84_train_sequential_conditional_gnn.py`：Sequential 基线。
4. `97_run_phase7_proposal_recovery_evaluation.py`：验证候选、必要时恢复、统计评估；`98_export_phase7_manuscript_evidence.py`：汇总已有完整评估产物。

历史训练器设有 `GDM_STAGE3_ACTIVE_ENTRY=1` 入口保护，并要求配置引用的数据冻结信息齐备。默认旧路径和阶段合同不能当作一条从零到论文表格的自动流水线；从头训练时需逐项准备对应配置、冻结清单和输出目录。当前公开版明确支持最小运行及源码检查，尚不提供完整论文资产的一键下载与复现。
