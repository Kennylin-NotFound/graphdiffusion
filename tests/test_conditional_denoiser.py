from pathlib import Path

import numpy as np
import pytest
import torch

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    all_masked_state,
)
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.models import (
    ConditionalDenoiserConfig,
    TypedFactorConditionalDenoiser,
)
from gdm_factor_diffusion.training import (
    MaskedConditionalTrainer,
    MaskedTrainerConfig,
    compute_masked_objective,
    load_masked_checkpoint,
    save_masked_checkpoint,
)


def _setup(device: str = "cpu"):
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance]).to(device)
    clean = torch.tensor([[1, 1, 2, 2, 2]], device=device)
    model = TypedFactorConditionalDenoiser.from_batch(
        batch,
        ConditionalDenoiserConfig(
            num_mask_steps=4,
            hidden_dim=32,
            num_layers=2,
        ),
    ).to(device)
    return batch, clean, model


def test_conditional_forward_backward_and_mask_invariants() -> None:
    batch, clean, model = _setup()
    partial = all_masked_state(batch.candidate_mask, batch.service_mask)
    logits = model(batch, partial, 4)
    terms = compute_masked_objective(logits, clean, partial, batch)
    terms.total.backward()

    assert logits.shape == batch.candidate_mask.shape
    assert torch.isneginf(logits[~batch.candidate_mask]).all()
    assert terms.hidden_count == int(batch.service_mask.sum())
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_partial_assignments_change_conditional_predictions() -> None:
    batch, clean, model = _setup()
    first_mask = torch.tensor([[True, False, False, False, False]])
    second_mask = torch.tensor([[False, True, False, False, False]])
    first = PartialPlacementState(clean.masked_fill(~first_mask, -1), first_mask)
    second = PartialPlacementState(clean.masked_fill(~second_mask, -1), second_mask)

    first_logits = model(batch, first, 3)
    second_logits = model(batch, second, 3)
    assert not torch.allclose(
        first_logits[batch.candidate_mask], second_logits[batch.candidate_mask]
    )


def test_masked_loss_ignores_visible_services() -> None:
    batch, clean, _ = _setup()
    committed = torch.tensor([[True, True, False, False, False]])
    partial = PartialPlacementState(clean.masked_fill(~committed, -1), committed)
    base = torch.zeros_like(batch.candidate_mask, dtype=torch.float32)
    changed = base.clone()
    changed[committed] = 1000.0

    first = compute_masked_objective(base, clean, partial, batch)
    second = compute_masked_objective(changed, clean, partial, batch)
    assert torch.equal(first.total, second.total)
    assert first.hidden_count == 3


def test_masked_trainer_and_checkpoint_round_trip(tmp_path: Path) -> None:
    batch, clean, model = _setup()
    schedule = AbsorbingMaskSchedule(num_steps=4)
    trainer = MaskedConditionalTrainer(
        model,
        schedule,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        MaskedTrainerConfig(gradient_clip_norm=1.0),
    )
    metrics = trainer.train_step(
        batch,
        clean,
        torch.tensor([3]),
        generator=torch.Generator().manual_seed(7),
    )
    saved = next(model.parameters()).detach().clone()
    path = save_masked_checkpoint(
        tmp_path / "masked.pt", trainer, metadata={"tag": "unit"}
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)
    metadata = load_masked_checkpoint(path, trainer)

    assert metadata == {"tag": "unit"}
    assert trainer.step == 1
    assert torch.equal(next(model.parameters()), saved)
    assert all(np.isfinite(value) for value in metrics.values())
    assert torch.load(path, weights_only=True)["model_kind"] == "masked_conditional"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_conditional_cuda_smoke() -> None:
    batch, clean, model = _setup("cuda")
    trainer = MaskedConditionalTrainer(
        model,
        AbsorbingMaskSchedule(num_steps=4),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
    )
    metrics = trainer.train_step(batch, clean, torch.tensor([4], device="cuda"))
    assert np.isfinite(metrics["loss_total"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_conditional_cuda_repeated_instance_batch() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance] * 8).to("cuda")
    model = TypedFactorConditionalDenoiser.from_batch(
        batch,
        ConditionalDenoiserConfig(
            num_mask_steps=4,
            hidden_dim=32,
            num_layers=2,
        ),
    ).to("cuda")
    partial = all_masked_state(batch.candidate_mask, batch.service_mask)

    logits = model(batch, partial, 4)

    assert logits.shape == batch.candidate_mask.shape
    assert torch.isfinite(logits[batch.candidate_mask]).all()
