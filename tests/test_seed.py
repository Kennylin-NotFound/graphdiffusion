import os
import random

import numpy as np
import torch

from gdm_factor_diffusion.common.seed import derive_seed, make_numpy_rng, seed_everything


def test_seed_everything_repeats_random_streams() -> None:
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_namespaced_numpy_streams_are_stable_and_distinct() -> None:
    assert derive_seed(5, "train") == derive_seed(5, "train")
    assert derive_seed(5, "train") != derive_seed(5, "test")
    assert np.array_equal(
        make_numpy_rng(5, "train").integers(0, 100, size=8),
        make_numpy_rng(5, "train").integers(0, 100, size=8),
    )


def test_deterministic_seed_configures_cublas_workspace() -> None:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    settings = seed_everything(19, deterministic=True)

    assert settings["cublas_workspace_config"] == ":4096:8"
