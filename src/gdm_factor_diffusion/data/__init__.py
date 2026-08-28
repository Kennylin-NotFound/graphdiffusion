"""Problem-instance schemas and dataset utilities."""

from .contracts import DISTRIBUTION_KEYS, audit_dataset_config_contract
from .schema import DeploymentInstance, load_instance, save_instance
from .toy import create_toy_instance
from .dataset import generate_dataset, load_manifest, load_manifest_instance, load_partition
from .generator import GeneratedInstance, InstanceGenerationSpec, generate_instance
from .graph_readiness import GraphReadinessReport, audit_graph_readiness
from .graph_blueprint import FactorGraphBlueprint, build_factor_graph_blueprint

__all__ = [
    "DeploymentInstance",
    "DISTRIBUTION_KEYS",
    "GeneratedInstance",
    "FactorGraphBlueprint",
    "GraphReadinessReport",
    "InstanceGenerationSpec",
    "audit_graph_readiness",
    "audit_dataset_config_contract",
    "build_factor_graph_blueprint",
    "create_toy_instance",
    "generate_dataset",
    "generate_instance",
    "load_instance",
    "load_manifest",
    "load_manifest_instance",
    "load_partition",
    "save_instance",
]
