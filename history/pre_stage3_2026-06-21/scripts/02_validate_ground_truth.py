"""Cross-check the deterministic evaluator and verifier on the toy instance."""

from __future__ import annotations

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.solver import enumerate_feasible_placements, evaluate_latency


def main() -> None:
    instance = create_toy_instance()
    result = enumerate_feasible_placements(instance)
    best = evaluate_latency(instance, result.best_placement)
    print(
        f"instance={instance.instance_id} candidates={result.num_candidates} "
        f"feasible={result.num_feasible}"
    )
    print(f"best_placement={result.best_placement.tolist()}")
    print(f"best_objective={result.best_objective:.12g}")
    print(f"application_latency={best.application_latency.tolist()}")
    print(f"completion_time={best.completion_time.tolist()}")


if __name__ == "__main__":
    main()
