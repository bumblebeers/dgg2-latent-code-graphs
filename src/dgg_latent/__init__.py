from dgg_latent.attn_fix import (
    install_dg_attn_fix,
    set_mask_variant,
)
from dgg_latent.losses import (
    PhaseController,
    balanced_slot_loss,
    distribution_ce,
    loss_hv,
    loss_pixel,
    per_example_reliability_weight,
)
from dgg_latent.readout_heads import (
    CrossSoftCanvasHead,
    DualReadoutSoftCanvasHead,
    ExpBInputAdapter,
    SoftCanvasHead,
    SymbolVectorInputAdapter,
    build_single_canvas_head,
    compute_readout_input,
)
from dgg_latent.sampling import (
    ContinuousLatentSampler,
    _GenRecorder,
    corrupt_discrete_canvas,
    cosine_corruption_schedule,
    linear_corruption_schedule,
    load_stock_generation_config,
    slerp,
    stock_generate,
    stock_generate_with_readout,
)

__version__ = "0.1.0"

__all__ = [
    "SoftCanvasHead",
    "DualReadoutSoftCanvasHead",
    "CrossSoftCanvasHead",
    "SymbolVectorInputAdapter",
    "ExpBInputAdapter",
    "compute_readout_input",
    "build_single_canvas_head",
    "distribution_ce",
    "loss_hv",
    "loss_pixel",
    "per_example_reliability_weight",
    "balanced_slot_loss",
    "PhaseController",
    "ContinuousLatentSampler",
    "corrupt_discrete_canvas",
    "linear_corruption_schedule",
    "cosine_corruption_schedule",
    "slerp",
    "stock_generate_with_readout",
    "stock_generate",
    "load_stock_generation_config",
    "_GenRecorder",
    "install_dg_attn_fix",
    "set_mask_variant",
]
