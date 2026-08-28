"""Run-directory, structured logging, and reproducibility metadata helpers."""

from __future__ import annotations

import json
import logging as py_logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)
    return destination


def create_run_directory(root: str | Path, name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / f"{timestamp}-{name}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{timestamp}-{name}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def configure_logging(
    run_directory: str | Path,
    level: int = py_logging.INFO,
) -> py_logging.Logger:
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    logger = py_logging.getLogger("gdm_factor_diffusion")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = py_logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console = py_logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = py_logging.FileHandler(run_path / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _package_versions() -> dict[str, str]:
    package_names = (
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "matplotlib",
        "PyYAML",
        "torch",
        "torch-geometric",
        "gurobipy",
    )
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_revision(project_root: str | Path | None) -> str | None:
    if project_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_run_metadata(
    seed: int,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    gpu: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_runtime": torch.version.cuda,
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config": config,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": platform.platform(),
        "packages": _package_versions(),
        "gpu": gpu,
        "git_revision": _git_revision(project_root),
    }
