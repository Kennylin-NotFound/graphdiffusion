"""Reproducible random-number utilities."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable 32-bit seed for an independent random stream."""

    payload = f"{int(base_seed)}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_numpy_rng(base_seed: int, namespace: str = "default") -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, namespace))


def seed_everything(seed: int, deterministic: bool = False) -> dict[str, object]:
    """Seed Python, NumPy, and PyTorch and return the applied settings."""

    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise ValueError("Seed must be nonnegative.")

    os.environ["PYTHONHASHSEED"] = str(normalized_seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)

    import torch

    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)

    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    return {
        "seed": normalized_seed,
        "deterministic": deterministic,
        "cuda_seeded": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
