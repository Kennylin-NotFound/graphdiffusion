"""Energy-weighted training utilities for the factor denoiser."""

from .data import (
    LabeledBatch,
    LabeledDeploymentDataset,
    LabeledItem,
    LabeledTarget,
    audit_dataset_freeze,
    make_labeled_collator,
    sample_clean_targets,
)
from .objectives import (
    ObjectiveTerms,
    capacity_guidance,
    clean_state_accuracy,
    compute_objective,
    link_guidance,
)
from .trainer import (
    DenoiserTrainer,
    TrainerConfig,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from .masked_objectives import MaskedObjectiveTerms, compute_masked_objective
from .masked_trainer import (
    MaskedConditionalTrainer,
    MaskedTrainerConfig,
    load_masked_checkpoint,
    restore_masked_checkpoint,
    save_masked_checkpoint,
)
from .sequential_objectives import (
    SequentialObjectiveTerms,
    build_teacher_forced_prefix,
    compute_sequential_objective,
)
from .sequential_trainer import (
    SequentialConditionalTrainer,
    SequentialTrainerConfig,
    load_sequential_checkpoint,
    restore_sequential_checkpoint,
    save_sequential_checkpoint,
)
from .production import (
    ConstrainedValidationConfig,
    capture_random_state,
    evaluate_constrained_validation,
    restore_random_state,
    sample_training_batch,
    validation_rank,
)
from .stage3_production import (
    Stage3SelectionConfig,
    evaluate_stage3_selection,
    stage3_selection_rank,
)

__all__ = [
    "DenoiserTrainer",
    "ConstrainedValidationConfig",
    "LabeledBatch",
    "LabeledDeploymentDataset",
    "LabeledItem",
    "LabeledTarget",
    "MaskedConditionalTrainer",
    "MaskedObjectiveTerms",
    "MaskedTrainerConfig",
    "ObjectiveTerms",
    "SequentialConditionalTrainer",
    "SequentialObjectiveTerms",
    "SequentialTrainerConfig",
    "TrainerConfig",
    "Stage3SelectionConfig",
    "audit_dataset_freeze",
    "build_teacher_forced_prefix",
    "capacity_guidance",
    "clean_state_accuracy",
    "compute_objective",
    "compute_masked_objective",
    "compute_sequential_objective",
    "link_guidance",
    "load_checkpoint",
    "load_masked_checkpoint",
    "load_sequential_checkpoint",
    "make_labeled_collator",
    "sample_clean_targets",
    "sample_training_batch",
    "capture_random_state",
    "evaluate_constrained_validation",
    "evaluate_stage3_selection",
    "restore_checkpoint",
    "restore_masked_checkpoint",
    "restore_sequential_checkpoint",
    "restore_random_state",
    "save_checkpoint",
    "save_masked_checkpoint",
    "save_sequential_checkpoint",
    "validation_rank",
    "stage3_selection_rank",
]
