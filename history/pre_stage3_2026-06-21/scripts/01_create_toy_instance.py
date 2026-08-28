"""Create, reload, and verify the canonical Phase 0 toy instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.common.config import get_config_value, load_config
from gdm_factor_diffusion.data import create_toy_instance, load_instance, save_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path from the toy configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config = load_config(
        [
            implementation_root / "configs" / "base.yaml",
            implementation_root / "configs" / "toy.yaml",
        ]
    )
    output = args.output or (
        implementation_root / get_config_value(config, "instance.output")
    )
    instance = create_toy_instance(get_config_value(config, "instance.id"))
    save_instance(instance, output)
    loaded = load_instance(output)
    if not instance.equivalent_to(loaded):
        raise RuntimeError("Toy-instance round trip changed the instance.")
    print(
        f"saved={output} services={loaded.num_services} devices={loaded.num_devices} "
        f"applications={loaded.num_applications} dependencies={loaded.num_dependencies}"
    )


if __name__ == "__main__":
    main()
