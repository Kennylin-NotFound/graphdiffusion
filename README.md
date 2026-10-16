# GraphDiffusion

使用图神经网络和掩码扩散模型求解微服务部署问题。项目将服务、设备及约束表示为因子图，生成部署方案后检查可行性，并尝试修复不可行的候选方案。

代码包含数据生成、模型训练和评估，以及 Direct GNN、Sequential GNN 和随机方法等基线。

## 安装

需要 Python 3.11，以下以 Windows PowerShell 为例：

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

Linux/macOS 使用 `source .venv/bin/activate`。使用 GPU 时，请安装与 CUDA 环境匹配的 PyTorch 版本。

## 跑一个小例子

生成一组包含 12 个场景的数据，再检查部署约束和图输入：

```bash
python scripts/03_generate_dataset.py --config configs/dataset_phase6e_e_controlled_shift_smoke.yaml --output artifacts/public_smoke
python scripts/04_audit_dataset.py artifacts/public_smoke
```

生成的数据保存在 `artifacts/public_smoke/`。基础测试可以这样运行：

```bash
python -m pytest tests/test_ground_truth.py tests/test_generator.py tests/test_graph_blueprint.py tests/test_recovery.py -q
```

## 训练和评估

模型代码在 `src/gdm_factor_diffusion/`，参数配置在 `configs/`，训练与评估脚本在 `scripts/`。其中 `97_run_phase7_proposal_recovery_evaluation.py` 用于评估，`98_export_phase7_manuscript_evidence.py` 用于汇总结果。

完整实验需要另外准备数据集和模型权重，具体步骤见[复现说明](docs/reproduction.md)。使用 MILP 生成标签时需要配置 Gurobi 许可证。
