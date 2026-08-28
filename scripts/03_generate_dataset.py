"""Generate a configured graph-ready dataset and print its partition summary."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.data import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dataset YAML file; defaults to the Phase 1B smoke configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional dataset output-root override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root / "configs" / "dataset_phase1b_smoke.yaml"
    )
    config = load_config(config_path)
    configured_output = Path(config["dataset"]["output"])
    output = args.output or (
        configured_output
        if configured_output.is_absolute()
        else implementation_root / configured_output
    )
    manifest = generate_dataset(config, output_root=output)
    counts = Counter(entry["partition"] for entry in manifest["instances"])
    print(f"dataset={manifest['dataset_name']} root={output}")
    print(f"instances={manifest['instance_count']} partitions={dict(counts)}")
    for partition in manifest["partitions"]:
        entries = [
            entry
            for entry in manifest["instances"]
            if entry["partition"] == partition
        ]
        service_counts = [entry["num_services"] for entry in entries]
        print(
            f"{partition}: count={len(entries)} "
            f"services=[{min(service_counts)},{max(service_counts)}]"
        )


if __name__ == "__main__":
    main()
