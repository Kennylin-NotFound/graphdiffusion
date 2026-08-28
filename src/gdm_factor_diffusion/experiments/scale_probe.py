"""End-to-end Phase 6B scaling measurements on a frozen labeled dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import (
    collect_run_metadata,
    create_run_directory,
    write_json,
)
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.data import build_factor_graph_blueprint, load_manifest
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.inference import InferenceConfig, solve_fallback_only, solve_with_model
from gdm_factor_diffusion.solver import MilpConfig, solve_milp
from gdm_factor_diffusion.training import (
    DenoiserTrainer,
    LabeledDeploymentDataset,
    TrainerConfig,
)

from .runtime import load_learned_solver
from .schema import file_sha256


@dataclass(frozen=True, slots=True)
class ScaleProbeConfig:
    seed: int = 20260618
    device: str = "cuda"
    milp_time_limit_seconds: float = 15.0
    milp_threads: int = 1
    training_profile_steps: int = 3
    inference_samples: int = 4
    inference_batch_size: int = 4
    inference_reverse_steps: int = 25
    repair_max_moves: int = 10
    fallback_max_search_nodes: int = 100_000

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("Scale-probe seed must be nonnegative.")
        if self.milp_time_limit_seconds <= 0 or self.milp_threads < 1:
            raise ValueError("MILP time limit and thread count must be positive.")
        if self.training_profile_steps < 1:
            raise ValueError("training_profile_steps must be positive.")
        if self.inference_samples < 1 or self.inference_batch_size < 1:
            raise ValueError("Inference sample counts must be positive.")


def _pool_reference(item) -> dict[str, Any]:
    records = item.pool.metadata.get("solve_records", ())
    first = next(
        (record for record in records if record.get("rank_generated") == 0),
        None,
    )
    placements = item.pool.placements
    pairwise_hamming = [
        float(np.mean(placements[left] != placements[right]))
        for left in range(item.pool.size)
        for right in range(left + 1, item.pool.size)
    ]
    return {
        "pool_size": item.pool.size,
        "pool_best": float(item.pool.latencies[0]),
        "pool_maximum": float(item.pool.latencies[-1]),
        "pool_spread": float(item.pool.latencies[-1] / item.pool.latencies[0] - 1.0),
        "pool_mean_pairwise_hamming_fraction": (
            mean(pairwise_hamming) if pairwise_hamming else 0.0
        ),
        "pool_generation_seconds": float(item.pool.metadata["elapsed_seconds"]),
        "pool_termination_reason": item.pool.metadata["termination_reason"],
        "pool_best_proven_optimal": bool(
            first is not None
            and first.get("optimal_under_exclusions")
            and abs(float(first["exact_objective"]) - float(item.pool.latencies[0]))
            <= 1e-8
        ),
    }


def _training_profile(
    items,
    *,
    loaded_solver,
    device: torch.device,
    config: ScaleProbeConfig,
) -> dict[str, Any]:
    instances = [item.instance for item in items]
    graph_start = perf_counter()
    batch = build_factor_graph_batch(
        instances,
        feature_schema=loaded_solver.feature_schema,
    ).to(device)
    graph_build_seconds = perf_counter() - graph_start
    clean_state = torch.full(batch.service_mask.shape, -1, dtype=torch.long)
    for index, item in enumerate(items):
        placement = torch.from_numpy(item.pool.placements[0])
        clean_state[index, : placement.numel()] = placement
    clean_state = clean_state.to(device)
    model = loaded_solver.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = DenoiserTrainer(
        model,
        loaded_solver.schedule,
        optimizer,
        TrainerConfig(),
    )
    generator = torch.Generator(device=device).manual_seed(
        derive_seed(config.seed, f"training-profile:{items[0].partition}")
    )
    warmup_timestep = torch.randint(
        1,
        loaded_solver.schedule.num_steps + 1,
        (batch.batch_size,),
        device=device,
        generator=generator,
    )
    trainer.train_step(batch, clean_state, warmup_timestep, generator=generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    metrics = None
    for _ in range(config.training_profile_steps):
        timestep = torch.randint(
            1,
            loaded_solver.schedule.num_steps + 1,
            (batch.batch_size,),
            device=device,
            generator=generator,
        )
        start = perf_counter()
        metrics = trainer.train_step(
            batch,
            clean_state,
            timestep,
            generator=generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(perf_counter() - start)
    assert metrics is not None
    return {
        "instances": len(items),
        "services": sum(item.instance.num_services for item in items),
        "candidate_edges": sum(
            int(item.instance.compatibility_mask.sum()) for item in items
        ),
        "graph_build_seconds": graph_build_seconds,
        "mean_training_step_seconds": mean(step_times),
        "maximum_training_step_seconds": max(step_times),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
        "last_step_metrics": metrics,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "instance_id",
        "partition",
        "num_services",
        "num_devices",
        "num_dependencies",
        "candidate_edges",
        "factor_relation_edges",
        "graph_build_seconds",
        "pool_size",
        "pool_spread",
        "pool_mean_pairwise_hamming_fraction",
        "pool_generation_seconds",
        "pool_best_proven_optimal",
        "milp_status",
        "milp_optimal",
        "milp_runtime_seconds",
        "milp_gap",
        "fallback_gap_to_pool_best",
        "fallback_total_seconds",
        "learned_gap_to_pool_best",
        "learned_total_seconds",
        "learned_raw_feasible_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})


def _partition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for partition in sorted({record["partition"] for record in records}):
        selected = [record for record in records if record["partition"] == partition]

        def average(key: str) -> float:
            values = [float(record[key]) for record in selected if record[key] is not None]
            return mean(values) if values else float("nan")

        summary[partition] = {
            "instances": len(selected),
            "services_min": min(record["num_services"] for record in selected),
            "services_max": max(record["num_services"] for record in selected),
            "services_mean": average("num_services"),
            "devices_min": min(record["num_devices"] for record in selected),
            "devices_max": max(record["num_devices"] for record in selected),
            "candidate_edges_mean": average("candidate_edges"),
            "factor_relation_edges_mean": average("factor_relation_edges"),
            "graph_build_seconds_mean": average("graph_build_seconds"),
            "pool_generation_seconds_mean": average("pool_generation_seconds"),
            "pool_size_mean": average("pool_size"),
            "pool_spread_mean": average("pool_spread"),
            "pool_mean_pairwise_hamming_fraction_mean": average(
                "pool_mean_pairwise_hamming_fraction"
            ),
            "pool_best_optimal_rate": average("pool_best_proven_optimal"),
            "milp_optimal_rate": average("milp_optimal"),
            "milp_runtime_seconds_mean": average("milp_runtime_seconds"),
            "fallback_gap_to_pool_best_mean": average("fallback_gap_to_pool_best"),
            "fallback_total_seconds_mean": average("fallback_total_seconds"),
            "learned_gap_to_pool_best_mean": average("learned_gap_to_pool_best"),
            "learned_total_seconds_mean": average("learned_total_seconds"),
            "learned_raw_feasible_rate_mean": average("learned_raw_feasible_rate"),
        }
    return summary


def run_scale_probe(
    dataset_root: str | Path,
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    config: ScaleProbeConfig | None = None,
    project_root: str | Path | None = None,
) -> Path:
    settings = config or ScaleProbeConfig()
    settings.validate()
    root = Path(dataset_root).resolve()
    freeze_path = root / "dataset_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Scale probe requires a fully audited frozen dataset.")
    checkpoint_path = Path(checkpoint).resolve()
    seed_everything(settings.seed)
    requested = torch.device(settings.device)
    device = (
        requested
        if requested.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    manifest = load_manifest(root / "manifest.json")
    partitions = tuple(manifest["partitions"])
    dataset = LabeledDeploymentDataset(root, partitions=partitions)
    loaded_solver = load_learned_solver(checkpoint_path, dataset, device)
    inference = InferenceConfig(
        num_samples=settings.inference_samples,
        sample_batch_size=settings.inference_batch_size,
        reverse_steps=settings.inference_reverse_steps,
        repair_max_moves=settings.repair_max_moves,
        fallback_max_search_nodes=settings.fallback_max_search_nodes,
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=True,
    )
    run_directory = create_run_directory(output_root, "phase6b-scale-probe")
    resolved = {
        "config": asdict(settings),
        "dataset_root": str(root),
        "dataset_freeze_sha256": file_sha256(freeze_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "device_requested": settings.device,
        "device_resolved": str(device),
        "partitions": list(partitions),
    }
    write_json(run_directory / "resolved_probe.json", resolved)
    write_json(
        run_directory / "run_meta.json",
        collect_run_metadata(settings.seed, resolved, project_root=project_root),
    )

    warmup_item = dataset[0]
    warmup_generator = torch.Generator(device=device).manual_seed(
        derive_seed(settings.seed, "scale-inference-warmup")
    )
    solve_with_model(
        loaded_solver.model,
        warmup_item.instance,
        loaded_solver.schedule,
        loaded_solver.feature_schema,
        config=inference,
        device=device,
        generator=warmup_generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        instance = item.instance
        graph_start = perf_counter()
        blueprint = build_factor_graph_blueprint(instance)
        build_factor_graph_batch(
            [instance],
            feature_schema=loaded_solver.feature_schema,
        )
        graph_seconds = perf_counter() - graph_start
        fallback = solve_fallback_only(
            instance,
            max_search_nodes=settings.fallback_max_search_nodes,
        )
        generator = torch.Generator(device=device).manual_seed(
            derive_seed(settings.seed, f"scale-learned:{instance.instance_id}")
        )
        learned = solve_with_model(
            loaded_solver.model,
            instance,
            loaded_solver.schedule,
            loaded_solver.feature_schema,
            config=inference,
            device=device,
            generator=generator,
        )
        milp = solve_milp(
            instance,
            MilpConfig(
                time_limit_seconds=settings.milp_time_limit_seconds,
                mip_gap=0.0,
                threads=settings.milp_threads,
                seed=settings.seed,
                output_flag=False,
            ),
        )
        reference = _pool_reference(item)
        pool_best = reference["pool_best"]
        record = {
            "instance_id": instance.instance_id,
            "partition": item.partition,
            "num_services": instance.num_services,
            "num_devices": instance.num_devices,
            "num_dependencies": instance.num_dependencies,
            "candidate_edges": int(instance.compatibility_mask.sum()),
            "factor_relation_edges": sum(
                relation.shape[1] for relation in blueprint.relation_index.values()
            ),
            "graph_build_seconds": graph_seconds,
            **reference,
            "milp_status": milp.status_name,
            "milp_optimal": milp.optimal,
            "milp_runtime_seconds": milp.runtime_seconds,
            "milp_gap": milp.mip_gap,
            "milp_exact_objective": milp.exact_objective,
            "fallback_gap_to_pool_best": (
                None
                if fallback.objective is None
                else fallback.objective / pool_best - 1.0
            ),
            "fallback_total_seconds": fallback.metrics["total_seconds"],
            "learned_gap_to_pool_best": (
                None
                if learned.objective is None
                else learned.objective / pool_best - 1.0
            ),
            "learned_total_seconds": learned.metrics["total_seconds"],
            "learned_raw_feasible_rate": learned.metrics["raw_feasible_rate"],
            "learned_final_success": learned.success,
        }
        records.append(record)
        print(
            f"instance={instance.instance_id} services={instance.num_services} "
            f"milp={milp.status_name}:{milp.runtime_seconds:.3f}s "
            f"pool={reference['pool_generation_seconds']:.3f}s "
            f"learned={learned.metrics['total_seconds']:.3f}s"
        )

    with (run_directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    _write_csv(run_directory / "records.csv", records)

    training_profiles = {}
    for partition in partitions:
        items = [dataset[index] for index in range(len(dataset)) if dataset[index].partition == partition]
        training_profiles[partition] = _training_profile(
            items,
            loaded_solver=load_learned_solver(checkpoint_path, dataset, device),
            device=device,
            config=settings,
        )
    summary = {
        "schema_version": "1.0",
        "run_directory": str(run_directory.resolve()),
        "records": len(records),
        "all_learned_outputs_verified": all(
            record["learned_final_success"] for record in records
        ),
        "partition_summary": _partition_summary(records),
        "training_profiles": training_profiles,
    }
    write_json(run_directory / "summary.json", summary)
    return run_directory


def export_scale_probe_figures(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Export scaling diagnostics from one completed probe."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_directory)
    records = [
        json.loads(line)
        for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(output_directory) if output_directory else run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    colors = {
        "seen_medium": "tab:blue",
        "large": "tab:orange",
        "extra_large": "tab:red",
    }
    generated = {}

    def save(figure, name: str) -> None:
        paths = []
        for suffix in (".png", ".pdf"):
            path = output / f"{name}{suffix}"
            figure.savefig(path, bbox_inches="tight", dpi=300)
            paths.append(str(path.resolve()))
        generated[name] = paths
        plt.close(figure)

    plt.rcParams.update({"font.size": 8, "figure.figsize": (3.5, 2.5)})
    figure, axis = plt.subplots()
    for partition, color in colors.items():
        selected = [record for record in records if record["partition"] == partition]
        axis.scatter(
            [record["num_services"] for record in selected],
            [record["learned_total_seconds"] for record in selected],
            label=partition.replace("_", " "),
            color=color,
            s=18,
        )
    axis.set_xlabel("Number of services")
    axis.set_ylabel("Learned online solving time (s)")
    axis.grid(True, linestyle=":", linewidth=0.5)
    axis.legend(frameon=False)
    save(figure, "inference_scaling")

    figure, axis = plt.subplots()
    for partition, color in colors.items():
        selected = [record for record in records if record["partition"] == partition]
        axis.scatter(
            [record["num_services"] for record in selected],
            [record["pool_generation_seconds"] for record in selected],
            label=partition.replace("_", " "),
            color=color,
            s=18,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Number of services")
    axis.set_ylabel("Four-solution pool time (s)")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(frameon=False)
    save(figure, "pool_scaling")

    figure, axis = plt.subplots()
    for partition, color in colors.items():
        selected = [record for record in records if record["partition"] == partition]
        axis.scatter(
            [record["num_services"] for record in selected],
            [100.0 * record["learned_gap_to_pool_best"] for record in selected],
            label=partition.replace("_", " "),
            color=color,
            s=18,
        )
    axis.set_xlabel("Number of services")
    axis.set_ylabel("Learned gap to exact optimum (%)")
    axis.grid(True, linestyle=":", linewidth=0.5)
    axis.legend(frameon=False)
    save(figure, "quality_scaling")

    figure, axis = plt.subplots()
    partitions = list(colors)
    raw = [
        100.0
        * mean(
            record["learned_raw_feasible_rate"]
            for record in records
            if record["partition"] == partition
        )
        for partition in partitions
    ]
    axis.bar(
        [partition.replace("_", " ") for partition in partitions],
        raw,
        color=[colors[partition] for partition in partitions],
    )
    axis.set_ylabel("Learned raw feasible rate (%)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5)
    save(figure, "raw_feasibility_by_scale")

    payload = {"schema_version": "1.0", "figures": generated}
    write_json(output / "figure_manifest.json", payload)
    return payload
