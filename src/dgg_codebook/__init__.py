"""dgg_codebook: Library for continuous vocabulary distribution derivation, de-biasing, and spatial teachers."""

from dgg_codebook.debiasing import (
    BackgroundSuppression,
    FrequencyDebiaser,
    RoPECarrierDebiaser,
)
from dgg_codebook.derivation import (
    apply_foreground_salience_boost,
    build_multimodal_target_distribution,
    compute_expected_vocabulary_embedding,
    sample_x0_from_distribution,
)
from dgg_codebook.teacher_models import (
    ContextualizedSpatialMLP,
    CtxSpatialTeacher,
    DistilledSpatialTeacher,
    IsoSpatialTeacher,
    SurrogateDistillationLoss,
    classify_lexical_token_class,
    extract_teacher_modes_from_hv_batched,
)

__version__ = "0.1.0"

__all__ = [
    "build_multimodal_target_distribution",
    "compute_expected_vocabulary_embedding",
    "apply_foreground_salience_boost",
    "sample_x0_from_distribution",
    "FrequencyDebiaser",
    "BackgroundSuppression",
    "RoPECarrierDebiaser",
    "IsoSpatialTeacher",
    "CtxSpatialTeacher",
    "DistilledSpatialTeacher",
    "ContextualizedSpatialMLP",
    "classify_lexical_token_class",
    "extract_teacher_modes_from_hv_batched",
    "SurrogateDistillationLoss",
]
