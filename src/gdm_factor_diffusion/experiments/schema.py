"""Validated configuration contract for paper-level experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from gdm_factor_diffusion.common.config import load_config, resolve_config_path

EXPERIMENT_SCHEMA_VERSION = "1.0"

METHOD_KINDS = {
    "learned_hybrid",
    "learned_repair",
    "learned_raw_only",
    "direct_hybrid",
    "direct_repair",
    "direct_raw_only",
    "random_hybrid",
    "random_repair",
    "random_raw_only",
    "fallback_only",
    "greedy_local",
    "latency_aware_heuristic",
    "local_search",
    "milp_time_limit",
}

TIMING_COMPONENTS = (
    "sampling_seconds",
    "optimization_seconds",
    "verification_seconds",
    "repair_seconds",
    "fallback_seconds",
    "exact_evaluation_seconds",
    "selection_seconds",
    "total_seconds",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_config_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    """One paper claim and the experiment evidence intended to test it."""

    claim_id: str
    question: str
    hypothesis: str
    comparison: tuple[str, ...]
    primary_metric: str

    def validate(self) -> None:
        if not self.claim_id.strip() or not self.question.strip():
            raise ValueError("Claim IDs and questions must be nonempty.")
        if len(self.comparison) < 1:
            raise ValueError("Each claim must name at least one method.")
        if not self.primary_metric.strip():
            raise ValueError("Each claim must name a primary metric.")


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One registered solver variant evaluated under a fixed budget."""

    method_id: str
    kind: str
    checkpoint: str | None = None
    proposal_group: str | None = None
    inference: dict[str, Any] = field(default_factory=dict)
    time_limit_seconds: float | None = None

    def validate(self) -> None:
        if not self.method_id.strip():
            raise ValueError("method_id must be nonempty.")
        if self.kind not in METHOD_KINDS:
            raise ValueError(f"Unsupported method kind: {self.kind!r}.")
        if self.kind.startswith(("learned_", "direct_")) and not self.checkpoint:
            raise ValueError(f"Method {self.method_id!r} requires a checkpoint.")
        if self.kind in {
            "fallback_only",
            "greedy_local",
            "latency_aware_heuristic",
            "local_search",
            "milp_time_limit",
        }:
            if self.checkpoint is not None:
                raise ValueError(f"{self.kind} must not specify a checkpoint.")
        if self.kind == "milp_time_limit" and self.time_limit_seconds is None:
            raise ValueError("milp_time_limit requires time_limit_seconds.")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive when specified.")
        allowed = {
            "num_samples",
            "sample_batch_size",
            "repair_max_moves",
            "fallback_max_search_nodes",
            "reverse_steps",
        }
        unknown = set(self.inference) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported inference settings for {self.method_id!r}: "
                f"{sorted(unknown)}"
            )
        if (
            self.kind
            in {
                "greedy_local",
                "latency_aware_heuristic",
                "local_search",
                "milp_time_limit",
            }
            and self.inference
        ):
            raise ValueError(f"{self.kind} does not accept inference settings.")

    @property
    def seed_namespace(self) -> str:
        return self.proposal_group or self.method_id


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Complete reproducibility contract for one evaluation run."""

    name: str
    dataset_root: str
    partitions: tuple[str, ...]
    methods: tuple[MethodSpec, ...]
    seed: int
    device: str
    output_root: str
    dataset_freeze: str = "dataset_freeze.json"
    instance_limit: int | None = None
    deterministic: bool = False
    timing_scope: tuple[str, ...] = TIMING_COMPONENTS
    claims: tuple[ClaimSpec, ...] = ()
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("Unsupported experiment manifest schema version.")
        if not self.name.strip() or not self.dataset_root.strip():
            raise ValueError("Experiment name and dataset_root must be nonempty.")
        if not self.partitions:
            raise ValueError("At least one evaluation partition is required.")
        if len(self.partitions) != len(set(self.partitions)):
            raise ValueError("Experiment partitions must be unique.")
        if self.seed < 0:
            raise ValueError("Experiment seed must be nonnegative.")
        if self.instance_limit is not None and self.instance_limit < 1:
            raise ValueError("instance_limit must be positive when provided.")
        method_ids = [method.method_id for method in self.methods]
        if not method_ids or len(method_ids) != len(set(method_ids)):
            raise ValueError("Experiment method IDs must be nonempty and unique.")
        for method in self.methods:
            method.validate()
        unknown_timing = set(self.timing_scope) - set(TIMING_COMPONENTS)
        if unknown_timing:
            raise ValueError(f"Unknown timing components: {sorted(unknown_timing)}")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Claim IDs must be unique.")
        for claim in self.claims:
            claim.validate()
            missing = set(claim.comparison) - set(method_ids)
            if missing:
                raise ValueError(
                    f"Claim {claim.claim_id!r} references unknown methods: "
                    f"{sorted(missing)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.to_dict())


def _claim_from_mapping(payload: Mapping[str, Any]) -> ClaimSpec:
    return ClaimSpec(
        claim_id=str(payload["claim_id"]),
        question=str(payload["question"]),
        hypothesis=str(payload["hypothesis"]),
        comparison=tuple(str(value) for value in payload["comparison"]),
        primary_metric=str(payload["primary_metric"]),
    )


def _method_from_mapping(payload: Mapping[str, Any]) -> MethodSpec:
    return MethodSpec(
        method_id=str(payload["method_id"]),
        kind=str(payload["kind"]),
        checkpoint=(
            None if payload.get("checkpoint") is None else str(payload["checkpoint"])
        ),
        proposal_group=(
            None
            if payload.get("proposal_group") is None
            else str(payload["proposal_group"])
        ),
        inference=dict(payload.get("inference", {})),
        time_limit_seconds=(
            None
            if payload.get("time_limit_seconds") is None
            else float(payload["time_limit_seconds"])
        ),
    )


def manifest_from_mapping(payload: Mapping[str, Any]) -> ExperimentManifest:
    manifest = ExperimentManifest(
        name=str(payload["name"]),
        dataset_root=str(payload["dataset_root"]),
        dataset_freeze=str(payload.get("dataset_freeze", "dataset_freeze.json")),
        partitions=tuple(str(value) for value in payload["partitions"]),
        methods=tuple(_method_from_mapping(value) for value in payload["methods"]),
        seed=int(payload["seed"]),
        device=str(payload.get("device", "cpu")),
        output_root=str(payload.get("output_root", "artifacts/experiments")),
        instance_limit=(
            None
            if payload.get("instance_limit") is None
            else int(payload["instance_limit"])
        ),
        deterministic=bool(payload.get("deterministic", False)),
        timing_scope=tuple(payload.get("timing_scope", TIMING_COMPONENTS)),
        claims=tuple(_claim_from_mapping(value) for value in payload.get("claims", ())),
        schema_version=str(
            payload.get("schema_version", EXPERIMENT_SCHEMA_VERSION)
        ),
    )
    manifest.validate()
    return manifest


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    config = load_config(path)
    payload = config.get("experiment", config)
    if not isinstance(payload, Mapping):
        raise TypeError("Experiment manifest must be a mapping.")
    return manifest_from_mapping(payload)
