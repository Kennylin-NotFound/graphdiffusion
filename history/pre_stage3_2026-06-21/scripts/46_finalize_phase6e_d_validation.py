"""Select and freeze the direct proposal budget from validation timing."""

from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ed_validation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = finalize_phase6ed_validation(
        root / "artifacts" / "phase6e-d-time-matched" / "validation_lock.json",
        implementation_root=root,
    )
    selected = result.get("selection", {}).get("selected_direct_k")
    print(f"status={result['status']} selected_direct_k={selected}")

