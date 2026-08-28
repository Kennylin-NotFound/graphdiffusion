"""Registered solver variants and shared checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from gdm_factor_diffusion.common.seed import derive_seed
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SolveResult,
    sample_random_proposals,
    solve_fallback_only,
    solve_from_proposals,
    solve_greedy_local,
    solve_latency_aware_heuristic,
    solve_local_search,
    solve_milp_time_limit,
    solve_with_direct_predictor,
    solve_with_model,
)
from gdm_factor_diffusion.models import (
    DenoiserConfig,
    DirectPredictorConfig,
    TypedFactorDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, LabeledItem

from .schema import MethodSpec


@dataclass(frozen=True, slots=True)
class LoadedLearnedSolver:
    model: torch.nn.Module
    schedule: CategoricalSchedule
    feature_schema: GraphFeatureSchema
    model_kind: str


def load_learned_solver(
    checkpoint_path: str | Path,
    dataset: LabeledDeploymentDataset,
    device: torch.device | str,
) -> LoadedLearnedSolver:
    target_device = torch.device(device)
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=target_device,
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    schema_payload = metadata["feature_schema"]
    feature_schema = GraphFeatureSchema(
        service_feature_names=tuple(schema_payload["service_feature_names"]),
        device_feature_names=tuple(schema_payload["device_feature_names"]),
        resource_names=tuple(schema_payload["resource_names"]),
    )
    reference = build_factor_graph_batch(
        [dataset[0].instance],
        feature_schema=feature_schema,
    ).to(target_device)
    saved_config = metadata["config"]
    model_config = saved_config["model"]
    model_kind = str(checkpoint.get("model_kind", "diffusion"))
    if model_kind == "diffusion":
        model = TypedFactorDenoiser.from_batch(
            reference,
            DenoiserConfig(
                num_diffusion_steps=int(saved_config["diffusion"]["steps"]),
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(target_device)
    elif model_kind == "direct":
        model = TypedFactorDirectPredictor.from_batch(
            reference,
            DirectPredictorConfig(
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(target_device)
    else:
        raise ValueError(f"Unsupported learned checkpoint model kind: {model_kind!r}.")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = CategoricalSchedule.from_betas(
        checkpoint["schedule_betas"].to(target_device)
    )
    return LoadedLearnedSolver(model, schedule, feature_schema, model_kind)


def inference_config_for_method(method: MethodSpec) -> InferenceConfig:
    values = {
        "num_samples": 4,
        "sample_batch_size": 4,
        "repair_max_moves": 10,
        "fallback_max_search_nodes": 100_000,
        "reverse_steps": 25,
    }
    values.update(method.inference)
    flags = {
        "learned_hybrid": (True, True, True),
        "random_hybrid": (True, True, True),
        "learned_repair": (True, False, False),
        "random_repair": (True, False, False),
        "learned_raw_only": (False, False, False),
        "direct_hybrid": (True, True, True),
        "direct_repair": (True, False, False),
        "direct_raw_only": (False, False, False),
        "random_raw_only": (False, False, False),
    }
    if method.kind == "fallback_only":
        config = InferenceConfig(
            fallback_max_search_nodes=int(values["fallback_max_search_nodes"])
        )
    else:
        enable_repair, enable_fallback, always_include_fallback = flags[method.kind]
        config = InferenceConfig(
            num_samples=int(values["num_samples"]),
            sample_batch_size=int(values["sample_batch_size"]),
            repair_max_moves=int(values["repair_max_moves"]),
            fallback_max_search_nodes=int(values["fallback_max_search_nodes"]),
            reverse_steps=(
                None
                if values.get("reverse_steps") is None
                else int(values["reverse_steps"])
            ),
            enable_repair=enable_repair,
            enable_fallback=enable_fallback,
            always_include_fallback=always_include_fallback,
        )
        if method.kind.startswith("direct_"):
            config = InferenceConfig(
                num_samples=config.num_samples,
                sample_batch_size=config.sample_batch_size,
                repair_max_moves=config.repair_max_moves,
                fallback_max_search_nodes=config.fallback_max_search_nodes,
                reverse_steps=None,
                enable_repair=config.enable_repair,
                enable_fallback=config.enable_fallback,
                always_include_fallback=config.always_include_fallback,
            )
    config.validate()
    return config


def run_registered_method(
    method: MethodSpec,
    item: LabeledItem,
    *,
    experiment_seed: int,
    device: torch.device,
    learned_solver: LoadedLearnedSolver | None,
) -> tuple[SolveResult, int]:
    instance = item.instance
    seed = derive_seed(
        experiment_seed,
        f"{method.seed_namespace}:{instance.instance_id}",
    )
    if method.kind == "greedy_local":
        return solve_greedy_local(instance), seed
    if method.kind == "latency_aware_heuristic":
        return solve_latency_aware_heuristic(instance), seed
    if method.kind == "local_search":
        return solve_local_search(instance), seed
    if method.kind == "milp_time_limit":
        assert method.time_limit_seconds is not None
        return (
            solve_milp_time_limit(
                instance,
                time_limit_seconds=method.time_limit_seconds,
                seed=seed,
                threads=1,
            ),
            seed,
        )
    config = inference_config_for_method(method)
    if method.kind == "fallback_only":
        return (
            solve_fallback_only(
                instance,
                max_search_nodes=config.fallback_max_search_nodes,
            ),
            seed,
        )
    if method.kind.startswith("learned_"):
        if learned_solver is None:
            raise ValueError(f"Method {method.method_id!r} has no loaded checkpoint.")
        if learned_solver.model_kind != "diffusion":
            raise ValueError(f"Method {method.method_id!r} requires a diffusion model.")
        generator = torch.Generator(device=device).manual_seed(seed)
        return (
            solve_with_model(
                learned_solver.model,
                instance,
                learned_solver.schedule,
                learned_solver.feature_schema,
                config=config,
                device=device,
                generator=generator,
            ),
            seed,
        )
    if method.kind.startswith("direct_"):
        if learned_solver is None:
            raise ValueError(f"Method {method.method_id!r} has no loaded checkpoint.")
        if learned_solver.model_kind != "direct":
            raise ValueError(f"Method {method.method_id!r} requires a direct model.")
        generator = torch.Generator(device=device).manual_seed(seed)
        return (
            solve_with_direct_predictor(
                learned_solver.model,
                instance,
                learned_solver.feature_schema,
                config=config,
                device=device,
                generator=generator,
            ),
            seed,
        )

    generator = torch.Generator().manual_seed(seed)
    start = perf_counter()
    proposals = sample_random_proposals(
        instance,
        num_samples=config.num_samples,
        generator=generator,
    )
    sampling_seconds = perf_counter() - start
    return (
        solve_from_proposals(
            instance,
            proposals,
            config=config,
            sampling_seconds=sampling_seconds,
            proposal_method="random_categorical",
        ),
        seed,
    )
