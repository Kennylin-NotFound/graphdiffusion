# Graph-Ready Synthetic Dataset Design

## Purpose

The generator creates synthetic instances that exactly match the deterministic
deployment formulation while exposing every array required by the future typed
factor graph. Generated data are problem instances, not experiment results.

## Constructive feasibility

Each instance is generated around a hidden witness placement:

1. application templates are merged into a joint DAG;
2. services are constructively assigned to hardware-compatible devices;
3. capacities are set with configurable slack above witness load;
4. witness devices are forced into the compatibility mask;
5. direct links required by witness dependencies are forced into the topology;
6. additional compatible choices and physical links are sampled;
7. the shared placement verifier confirms feasibility.

The witness is saved only in `manifest.json` for generation auditing. It is not
stored in the instance NPZ and is explicitly marked as not being model input.
It is neither an optimal solution nor a training label.

## Safe microservice sharing

Two local services may be merged only when both their stable service type and
their actual global predecessor set match. This prevents the joint DAG from
merging services that have different inputs, configurations, or upstream
execution paths. Lower share probability creates semantically equivalent
replicas rather than unsafe merges.

## Partitions

- `train`: in-distribution seen-size instances used for solution-pool labels
  and model fitting.
- `validation`: disjoint in-distribution instances used for model selection.
- `test_id`: disjoint in-distribution instances used for final quality tests.
- `test_unseen_size`: more applications and devices than training.
- `test_sparse_topology`: lower requested physical-link density.
- `test_low_compatibility`: fewer device candidates per service.
- `test_high_sharing`: a higher probability of safe upstream reuse.
- `test_unseen_workflow`: uses application templates excluded from all
  training, validation, and in-distribution test partitions.

The committed smoke configuration reserves application template `5`
(`branched_detection`) for `test_unseen_workflow`; the remaining partitions
sample only from template IDs `0` through `4`. The unseen-size partition uses
the training template pool so that size and workflow-structure shifts remain
separable. The unseen-workflow partition uses two applications and elevated
safe sharing to keep its realized joint-DAG size near the seen-size range.

## Planned claim-to-test matrix

| Planned question | Partition or intervention | Primary diagnostics |
|---|---|---|
| Does the learned solver work on unseen in-distribution instances? | `test_id` | objective gap, feasibility rate, runtime |
| Does it generalize to larger graphs? | `test_unseen_size` | quality and runtime versus graph size |
| Does it handle limited communication choices? | `test_sparse_topology` | feasibility, repair rate, objective gap |
| Does it handle restricted deployment choices? | `test_low_compatibility` | feasibility, failure rate, candidate count |
| Does it use shared-workflow structure? | `test_high_sharing` | quality versus realized sharing ratio |
| Does it generalize to held-out DAG structure? | `test_unseen_workflow` | quality on template IDs absent from training |

All entries are planned experiment protocols, not current evidence.

Instance IDs and generation seeds are unique across all partitions. Solution
pools must later be generated and stored per instance; they must never cross
partitions.

Each NPZ file has a SHA-256 digest in `manifest.json`. The partition loader
verifies this digest by default so corrupted or silently modified instances
cannot enter training or evaluation.

## Size scaling

The committed smoke configuration uses small counts to validate the complete
pipeline quickly. Future full experiments should scale partition counts while
keeping the same regime definitions and should report actual service,
dependency, candidate-edge, and physical-link counts from the manifest rather
than relying only on requested ranges.

Do not fix the final full-dataset counts before Phase 1C measures MILP and
solution-pool throughput. A sensible progression is smoke validation, a pilot
dataset used to estimate labeling time and storage, and only then the final
multi-seed dataset.

## Graph input contract

The future graph builder receives only explicit instance arrays:

- service nodes: `service_type_id`, `service_features`, and dynamic state;
- device nodes: `device_type_id`, `device_features`, and capacities;
- dependency factors: `dependency_index`, finite pair costs, and admissibility;
- application factors: `application_type_id`, weights, membership, and sinks;
- categorical choice edges: `compatibility_mask`;
- physical-topology edges: `connectivity` and `link_rate`.

`audit_graph_readiness` checks these relations and dimensions before an
instance is accepted into a dataset.

`build_factor_graph_blueprint` constructs the framework-independent typed
relation indices and aligned static edge attributes immediately during Phase
1B. The later PyG graph builder should translate this blueprint rather than
re-derive relations from metadata.

## Current evidence boundary

The committed catalog uses normalized synthetic compute, data, link-rate, and
resource units. It satisfies the equations and constraints in Section II, but
it is not calibrated from measured video-analytics traces or real device
specifications. Before revising Section V, either calibrate these distributions
from defensible sources or describe the experiments explicitly as normalized
synthetic workloads. The current experiment prose must not present these
values as measured GHz, GB, bandwidth, or latency.
