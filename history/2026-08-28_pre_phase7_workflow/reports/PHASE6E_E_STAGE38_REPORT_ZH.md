# Phase 6E-E Stage 3.8 Sealed Confirmation Report

Updated: 2026-06-23

## 结论

Stage 3.8 已完成，但预注册的严格确认门禁没有通过。`final_decision_lock.json`
中记录为 `diffusion_claim_confirmed=false`。因此，论文中不能写成“图扩散在相近
在线预算下已被严格证明优于 Direct predictor”。

更稳妥的结论是：Masked Diffusion K=8 在聚合指标上相对 Direct K=64 呈现更好的
质量与可行性趋势，并且在线时间处于同一量级，但三种子 sealed confirmation 还不足
以支持强统计确认。

## 完成的工作

- 三个种子 `2026070111`--`2026070113` 的训练已冻结。
- 128 个 fresh sealed ID 实例已生成、Gurobi 标注、audit 并冻结。
- 每个 sealed 实例包含 16 个 MILP solution-pool placements，共 2048 个标注解。
- 三种方法完成 384 条 sealed 评估记录：
  `direct_k64`、`masked_deterministic_k1`、`masked_diffusion_k8`。
- 完整测试通过：`135 passed`。

## 关键证据

| Method | Final gap | Pre-fallback gap | Raw feasible | Total time |
|---|---:|---:|---:|---:|
| Direct K=64 | 1.631% | 3.693% | 48.96% | 0.501 s |
| Masked deterministic K=1 | 1.589% | 4.020% | 94.53% | 0.305 s |
| Masked Diffusion K=8 | 1.254% | 2.912% | 95.57% | 0.486 s |

Masked Diffusion K=8 的平均 final gap 和 pre-fallback gap 均优于 Direct K=64，
且平均总耗时略低于 Direct K=64 的锁定预算。然而严格门禁要求所有关键检查都通过。

## 未通过的门禁

- `positive_gap_improvement_each_seed=false`：种子 `2026070112` 的
  pre-fallback gap 相对 Direct K=64 变差。
- `instance_sign_test_passes=false`：实例级 paired sign test 为 70 wins、
  51 losses、7 ties，p = 0.101，没有达到显著性门槛。

## 论文写作边界

可以写：

- absorbing-MASK diffusion improves aggregate verified latency and raw proposal
  feasibility under the locked comparable-time evaluation.
- the improvement is promising but not uniformly significant across all seeds.
- hard verification and fallback remain necessary for final feasibility.

不要写：

- the proposed diffusion solver is statistically confirmed to dominate Direct
  prediction under matched online time.
- diffusion itself guarantees feasible deployments.
- final 100% success is evidence of neural proposal feasibility.

## 证据文件

- Final evidence:
  `implementation/artifacts/phase6e-e-stage38-sealed-evaluation/final_evidence.json`
- Final evidence SHA-256:
  `DDD1E46EEA36B45A333C6DEB26148F5C2A74FB2D5A03FF8C8598D0201DB0D636`
- Decision lock:
  `implementation/artifacts/phase6e-e-stage38/final_decision_lock.json`
- Decision lock SHA-256:
  `9C9CC6A20F71A4D8F9EB93EFBD8490C141119345684B5FA0A1085E5F5ED9EF96`
