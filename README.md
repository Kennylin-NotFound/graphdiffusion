# GraphDiffusion · 图条件掩码扩散微服务部署

面向资源和通信约束下微服务部署的研究代码。系统以类型化因子图表示服务、设备及约束，通过图条件掩码扩散生成部署方案，再执行硬约束验证；当原始候选均不可行时，尝试由候选引导的有界恢复，并用精确应用时延选择可行解。

当前方法入口为 Phase 7，包含 Direct GNN、Sequential GNN、Masked Diffusion 和 Random + Recovery 等比较路径。代码包名为 `gdm_factor_diffusion`。

## 环境安装

要求 Python **3.11**。在仓库根目录执行（PowerShell）：

```powershell
git clone https://github.com/Kennylin-NotFound/graphdiffusion.git
cd graphdiffusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase0.txt
python -m pip install torch==2.7.1
python -m pip install -r requirements-ml.txt
python -m pip install -e .
```

Linux/macOS 使用 `source .venv/bin/activate`。CUDA 训练需按机器和驱动选择对应的 PyTorch wheel；`requirements-ml.txt` 记录了原 CUDA 安装命令。MILP 标注需要有效的 Gurobi 许可证；下方数据生成和基础验证不求解 MILP。

## 最小复现：数据生成与验证

```bash
python scripts/03_generate_dataset.py --config configs/dataset_phase6e_e_controlled_shift_smoke.yaml --output artifacts/public_smoke
python scripts/04_audit_dataset.py artifacts/public_smoke
python -m pytest tests/test_ground_truth.py tests/test_generator.py tests/test_graph_blueprint.py tests/test_recovery.py -q
```

该示例生成 12 个小型场景，检查见证部署、约束与图输入，并验证时延和候选恢复逻辑。输出只写入被忽略的 `artifacts/public_smoke/`。它验证基础流程，不代表完整论文实验复现或论文指标。

## 论文实验复现

完整评估入口为 `scripts/97_run_phase7_proposal_recovery_evaluation.py`，结果汇总入口为 `scripts/98_export_phase7_manuscript_evidence.py`。训练实现、配置、评估与导出源码均保留。

**当前 GitHub 版不包含冻结数据集、训练权重、MILP 参考池和历史实验结果，因此仅 clone 仓库不能直接复现论文的完整数值表格。** 所需目录、执行命令和训练入口见 [复现说明](docs/reproduction.md)。论文 LaTeX、导师意见与投稿材料位于独立本地工作区，不属于此代码仓库。

## 目录

| 路径 | 内容 |
|---|---|
| `src/gdm_factor_diffusion/` | 数据、图模型、扩散、训练、推理、恢复和评估 |
| `configs/` | 数据集、训练和评估配置 |
| `scripts/` | 数据生成、标注、训练及 Phase 7 评估入口 |
| `tests/` | 算法回归测试；部分历史测试另需本地冻结数据 |
| `docs/reproduction.md` | 最小运行和完整实验的依赖说明 |
| `DATASET_DESIGN.md` / `SCHEMA.md` | 数据设计和格式 |

`PHASE6E_E_STAGE3_PLAN.md` 是训练合同哈希的一部分，因代码依赖而保留。`history/`、本机调度脚本、缓存、日志和 `artifacts/` 留在本地，不进入公开版本。不要直接把全量 `pytest` 作为无数据 clone 的验收命令；优先执行上方不依赖冻结产物的测试。
