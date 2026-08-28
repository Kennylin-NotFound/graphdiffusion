"""Immutable instance and solution-pool loading for denoiser training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from gdm_factor_diffusion.data.dataset import load_manifest, load_manifest_instance
from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.graph.batch_adapter import (
    FactorGraphBatch,
    GraphFeatureSchema,
    build_factor_graph_batch,
    infer_feature_schema,
)
from gdm_factor_diffusion.solver.solution_pool import SolutionPool, load_solution_pool


@dataclass(frozen=True, slots=True)
class LabeledItem:
    instance: DeploymentInstance
    pool: SolutionPool
    partition: str


@dataclass(frozen=True, slots=True)
class LabeledBatch:
    items: tuple[LabeledItem, ...]
    factor_graph: FactorGraphBatch


@dataclass(frozen=True, slots=True)
class LabeledTarget:
    state: Tensor
    pool_index: Tensor
    latency: Tensor
    normalized_energy: Tensor


def audit_dataset_freeze(dataset_root: str | Path) -> dict:
    """Verify that a labeled dataset still matches its immutable freeze hashes."""

    root = Path(dataset_root)
    freeze_path = root / "dataset_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(f"Frozen dataset contract is missing: {freeze_path}")
    with freeze_path.open("r", encoding="utf-8") as stream:
        freeze = json.load(stream)
    for name, expected in freeze["core_sha256"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Frozen dataset core hash mismatch: {name}")
    return freeze


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LabeledDeploymentDataset(Dataset[LabeledItem]):
    """Lazily load checksum-protected instances and their audited solution pools."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        partitions: Sequence[str],
        verify_checksum: bool = True,
        require_freeze: bool = False,
    ) -> None:
        self.root = Path(dataset_root)
        self.verify_checksum = verify_checksum
        if require_freeze:
            audit_dataset_freeze(self.root)
        manifest = load_manifest(self.root / "manifest.json")
        unknown = set(partitions) - set(manifest["partitions"])
        if unknown:
            raise ValueError(f"Unknown dataset partitions: {sorted(unknown)}")
        with (self.root / "solution_pool_manifest.json").open(
            "r", encoding="utf-8"
        ) as stream:
            pool_manifest = json.load(stream)
        pool_by_instance = {
            entry["instance_id"]: entry for entry in pool_manifest["pools"]
        }
        self.entries: list[tuple[dict, dict]] = []
        selected = set(partitions)
        for entry in manifest["instances"]:
            if entry["partition"] not in selected:
                continue
            pool_entry = pool_by_instance.get(entry["instance_id"])
            if pool_entry is None:
                raise ValueError(
                    f"Missing solution pool for instance {entry['instance_id']!r}."
                )
            if pool_entry["instance_sha256"] != entry["sha256"]:
                raise ValueError(
                    f"Stale solution pool for instance {entry['instance_id']!r}."
                )
            self.entries.append((entry, pool_entry))
        if not self.entries:
            raise ValueError("No labeled instances were selected.")

        schema_instances = [
            load_manifest_instance(
                self.root,
                entry,
                verify_checksum=self.verify_checksum,
            )
            for entry, _ in self.entries
        ]
        self.feature_schema = infer_feature_schema(schema_instances)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> LabeledItem:
        instance_entry, pool_entry = self.entries[index]
        instance = load_manifest_instance(
            self.root,
            instance_entry,
            verify_checksum=self.verify_checksum,
        )
        pool_path = self.root / pool_entry["pool_path"]
        if self.verify_checksum and _sha256(pool_path) != pool_entry["pool_sha256"]:
            raise ValueError(f"Checksum mismatch for solution pool: {pool_path}")
        pool = load_solution_pool(pool_path)
        if pool.instance_id != instance.instance_id:
            raise ValueError("Loaded solution pool and instance IDs disagree.")
        if pool.placements.shape[1] != instance.num_services:
            raise ValueError("Loaded solution pool has the wrong service dimension.")
        return LabeledItem(
            instance=instance,
            pool=pool,
            partition=instance_entry["partition"],
        )


def make_labeled_collator(
    feature_schema: GraphFeatureSchema,
) -> Callable[[Sequence[LabeledItem]], LabeledBatch]:
    """Create a picklable-style closure that enforces one feature schema."""

    def collate(items: Sequence[LabeledItem]) -> LabeledBatch:
        if not items:
            raise ValueError("Cannot collate an empty labeled batch.")
        immutable_items = tuple(items)
        factor_graph = build_factor_graph_batch(
            [item.instance for item in immutable_items],
            feature_schema=feature_schema,
        )
        return LabeledBatch(items=immutable_items, factor_graph=factor_graph)

    return collate


def sample_clean_targets(
    batch: LabeledBatch,
    *,
    mode: str = "energy",
    generator: torch.Generator | None = None,
) -> LabeledTarget:
    """Sample one verified clean placement per instance from its solution pool."""

    if mode not in {"energy", "uniform", "best"}:
        raise ValueError("mode must be 'energy', 'uniform', or 'best'.")
    graph = batch.factor_graph
    state = torch.full(graph.service_mask.shape, -1, dtype=torch.long)
    pool_index = torch.empty(graph.batch_size, dtype=torch.long)
    latency = torch.empty(graph.batch_size, dtype=torch.float64)
    normalized_energy = torch.empty(graph.batch_size, dtype=torch.float64)
    for batch_index, item in enumerate(batch.items):
        if mode == "best":
            selected = 0
        else:
            probability = (
                torch.from_numpy(item.pool.sampling_probability)
                if mode == "energy"
                else torch.full((item.pool.size,), 1.0 / item.pool.size)
            )
            selected = int(
                torch.multinomial(
                    probability,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).item()
            )
        placement = torch.from_numpy(item.pool.placements[selected])
        state[batch_index, : placement.numel()] = placement
        pool_index[batch_index] = selected
        latency[batch_index] = float(item.pool.latencies[selected])
        normalized_energy[batch_index] = float(
            item.pool.normalized_energy[selected]
        )
    return LabeledTarget(
        state=state,
        pool_index=pool_index,
        latency=latency,
        normalized_energy=normalized_energy,
    )
