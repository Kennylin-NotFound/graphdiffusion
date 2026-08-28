import json
from pathlib import Path

from gdm_factor_diffusion.common.logging import (
    collect_run_metadata,
    configure_logging,
    create_run_directory,
    write_json,
)


def test_run_logging_and_metadata(tmp_path: Path) -> None:
    run_directory = create_run_directory(tmp_path, "toy")
    logger = configure_logging(run_directory)
    logger.info("phase0-test")

    metadata = collect_run_metadata(seed=3, config={"runtime": {"seed": 3}})
    metadata_path = write_json(run_directory / "run_meta.json", metadata)
    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert loaded["seed"] == 3
    assert loaded["packages"]["torch"] != "not-installed"
    assert "phase0-test" in (run_directory / "run.log").read_text(encoding="utf-8")
