"""Cross-check the Phase 1C MILP against bounded exhaustive enumeration."""

from __future__ import annotations

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    create_toy_instance,
    generate_instance,
)
from gdm_factor_diffusion.solver import (
    MilpConfig,
    enumerate_feasible_placements,
    solve_milp,
)


def _generated_instance(seed: int):
    return generate_instance(
        InstanceGenerationSpec(
            instance_id=f"phase1c-check-{seed}",
            seed=seed,
            partition="validation",
            role="validation",
            regime="milp_cross_check",
            size_profile="tiny",
            num_applications=1,
            num_devices=4,
            share_probability=0.0,
            compatibility_density=0.7,
            topology_density=0.7,
            capacity_slack=0.6,
            minimum_candidates=2,
            application_type_ids=(2,),
        )
    ).instance


def main() -> None:
    instances = [create_toy_instance()] + [
        _generated_instance(seed) for seed in (101, 202, 303)
    ]
    for instance in instances:
        exhaustive = enumerate_feasible_placements(instance, max_states=10_000)
        result = solve_milp(instance, MilpConfig(output_flag=False))
        if not result.optimal:
            raise RuntimeError(f"MILP was not solved optimally: {result}")
        if abs(float(result.exact_objective) - exhaustive.best_objective) > 1e-8:
            raise RuntimeError(f"MILP/exhaustive mismatch: {instance.instance_id}")
        print(
            f"{instance.instance_id}: candidates={exhaustive.num_candidates} "
            f"feasible={exhaustive.num_feasible} "
            f"objective={result.exact_objective:.12g} "
            f"runtime={result.runtime_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()
