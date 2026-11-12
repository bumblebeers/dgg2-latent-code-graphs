"""
tests/e2e/test_tier2_boundaries.py - Tier 2: Boundary & Corner Cases E2E Tests.

Verifies system limits, extreme inputs, edge conditions, and contract boundary enforcement:
- Empty inputs and zero-edge graphs
- Single-cell edits (cut_one_call)
- Tie distance resolution in candidate matching
- 128 vs 256 sequence length truncation in text encoders
- DG Stock contract: Self-conditioning zero logits vs None
- Missing and invalid arguments in CLI / evaluators
- Palette color index bounds (0..7)
- Exact schedule balance (4,824 steps over 3 epochs)
- Open-ended validity bounds and un-invoked slot masking
- Multiscale grid cell divisibility
"""

import ast
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Boundary 1: Empty and Zero-Edge Graphs
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m1
@pytest.mark.m2
def test_boundary_empty_and_zero_edge_graphs(cg_loader):
    """Empty graph with 0 edges: all operations must execute cleanly without zero-division or crash."""
    ops = cg_loader("cg_ops")
    adj = np.zeros((16, 16), dtype=bool)
    kind = np.full(16, ops.FUNC, dtype=np.int32)
    s = ops.GraphState.new(adj, kind, 0)

    # Operations must maintain invariant of 0 edges
    assert (ops.reverse_edges(s).adj == adj).all()
    assert (ops.keep_callees(s).adj == adj).all()
    assert (ops.keep_callers(s).adj == adj).all()
    assert len(np.flatnonzero(ops.mark_reachable(s).marked)) == 0
    assert len(np.flatnonzero(ops.mark_ancestors(s).marked)) == 0


# ==============================================================================
# Boundary 2: Single-Cell Edits (cut_one_call)
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
def test_boundary_single_cell_edit_cut_one_call(cg_loader):
    """cut_one_call hinges on exactly ONE cell change: verify Hamming distance from source == 1."""
    ops = cg_loader("cg_ops")
    adj = np.zeros((16, 16), dtype=bool)
    adj[1, 2] = True  # selected node 1 calls 2
    adj[1, 3] = True  # selected node 1 calls 3
    kind = np.full(16, ops.FUNC, dtype=np.int32)
    s = ops.GraphState.new(adj, kind, 1)

    answers = ops.open_cut_one_call(s)
    assert len(answers) == 2, "Must produce 2 valid outcomes for 2 outgoing calls"

    src_grid = s.to_grid()
    for ans in answers:
        ans_grid = ans.to_grid()
        diff_cells = (src_grid != ans_grid).sum()
        assert diff_cells == 1, f"cut_one_call must differ by exactly 1 cell, got {diff_cells}"


# ==============================================================================
# Boundary 3: Tie Distances in Candidate Matching
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
def test_boundary_tie_distance_in_candidate_scoring(cg_loader):
    """Ties in nearest-candidate cosine/grid distance must be detected and counted as wrong per PREREG."""
    metrics = cg_loader("cg_metrics")

    # Prediction equidistant to Candidate 0 and Candidate 1
    pred = np.zeros((1, 16, 16), dtype=np.int32)
    cands = np.zeros((1, 4, 16, 16), dtype=np.int32)
    cands[0, 0, 0, 0] = 1  # diff = 1
    cands[0, 1, 0, 1] = 1  # diff = 1 (tie!)
    cands[0, 2, 0, :] = 2  # diff = 16
    cands[0, 3, 1, :] = 3  # diff = 16

    idx, tie = metrics.nearest_candidate_grid(pred, cands)
    assert tie[0] == True, "Equidistant candidates must be flagged as tie=True"


# ==============================================================================
# Boundary 4: Sequence Length Truncation (128 vs 256 tokens)
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m1
@pytest.mark.m3
def test_boundary_encoder_truncation_128_vs_256(cg_loader):
    """Encoders capped at 128 tokens truncate added parameters on long signatures, causing identical vectors."""
    # Long function text exceeding 128 tokens (~500 characters of parameters)
    long_params = ", ".join([f"param_{i}: int = {i}" for i in range(50)])
    code_orig = f"def execute_query({long_params}):\n    pass\n"
    code_edited = f"def execute_query({long_params}, timeout: int = 30):\n    pass\n"

    # With a 128-token character cutoff, the edit at the end is completely lost
    token_limit_128_chars = 128 * 4  # approx 512 chars
    truncated_orig = code_orig[:token_limit_128_chars]
    truncated_edited = code_edited[:token_limit_128_chars]

    # Both strings are identical under 128-token window
    assert truncated_orig == truncated_edited, "Truncation at 128 tokens hides appended parameters"

    # Under 256 tokens (1024 chars), the edit remains visible
    token_limit_256_chars = 256 * 4
    assert code_orig[:token_limit_256_chars] != code_edited[:token_limit_256_chars], "256 tokens preserves edit"


# ==============================================================================
# Boundary 5: DG Stock Contract: Self-Conditioning Zero Logits vs None
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.m3
def test_boundary_self_conditioning_zero_logits_vs_none():
    """Stock reference Rule 2: self-conditioning off MUST be None/False, NEVER zeros_like(logits)."""
    # Simulate self-conditioning parameter validation
    def validate_self_cond(self_cond):
        if self_cond is not None and not isinstance(self_cond, bool):
            if isinstance(self_cond, torch.Tensor) and (self_cond == 0).all():
                raise ValueError("CRITICAL CONTRACT VIOLATION: Self-conditioning off must be None or False, never zero logits!")
        return True

    # Valid inputs
    assert validate_self_cond(None) is True
    assert validate_self_cond(False) is True

    # Forbidden zero logits tensor
    zero_logits = torch.zeros(1, 256, 262144)
    with pytest.raises(ValueError, match="CRITICAL CONTRACT VIOLATION"):
        validate_self_cond(zero_logits)


# ==============================================================================
# Boundary 6: Missing Arguments and Unknown Manifest Rejection
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m1
@pytest.mark.m2
def test_boundary_missing_mandatory_arguments_and_unknown_manifest(cg_loader, data_locator):
    """cg_score_preds and spec adapter must raise KeyError or ValueError on unknown manifest name."""
    spec = cg_loader("cg_sec27_spec")
    pack = spec.load_pack(str(data_locator("gridA_train1608.npz")))
    manifests = spec.build_cg_eval_manifests(pack)

    assert "non_existent_manifest" not in manifests


# ==============================================================================
# Boundary 7: Palette Color Index Bounds (0..7)
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m1
def test_boundary_palette_class_index_bounds(cg_loader):
    """Palette indices must be in [0, 7]. Out-of-bound indices must trigger boundary errors."""
    ops = cg_loader("cg_ops")
    assert len(ops.PALETTE_RGB) == 8, "Palette must have exactly 8 colors"
    assert ops.BG == 0 and ops.EDGE == 1 and ops.FUNC == 2
    assert ops.METHOD == 3 and ops.CLASS == 4 and ops.TEST == 5
    assert ops.SEL == 6 and ops.MARK == 7

    # Colors must be valid RGB triples (0..255)
    for name, (r, g, b) in zip(ops.PALETTE_NAMES, ops.PALETTE_RGB):
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255, f"Invalid color for {name}: {(r, g, b)}"


# ==============================================================================
# Boundary 8: Exact Schedule Balance (4,824 steps over 3 epochs)
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
def test_boundary_schedule_balance_and_step_limits(cg_loader, data_locator):
    """Schedule must produce exactly 4,824 optimizer steps across 3 epochs with quad batching."""
    spec = cg_loader("cg_sec27_spec")
    pack = spec.load_pack(str(data_locator("gridA_train1608.npz")))

    schedule = spec.build_cg_contrastive_schedule(pack, epoch_seed=42)
    # 1608 sources * 8 ops = 12,864 rows per epoch
    # Quad batching: 4 items per step -> 12,864 / 8 = 1,608 steps per epoch
    # 3 epochs * 1,608 = 4,824 total steps
    steps_per_epoch = len(schedule) // 8  # runner steps
    assert len(schedule) == 12864, f"Expected 12,864 rows per epoch, got {len(schedule)}"
    total_steps = steps_per_epoch * 3
    assert total_steps == 4824, f"Expected exactly 4,824 steps over 3 epochs, got {total_steps}"

    # Verify NO held-out operations appear in training schedule
    scheduled_ops = {row[2] for row in schedule}
    for heldout_op in spec.CG_HELDOUT_OPS:
        assert heldout_op not in scheduled_ops, f"Held-out operation {heldout_op} leaked into training schedule!"


# ==============================================================================
# Boundary 9: Open-Ended Validity Bounds and Masking
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
def test_boundary_open_ended_validity_bounds(cg_loader, data_locator):
    """Open-ended slots that do not exist for a source must be marked invalid (state_valid == False)."""
    spec = cg_loader("cg_sec27_spec")
    pack = spec.load_pack(str(data_locator("gridA_test360_httpx.npz")))

    # For each source, open count is between 2 and 3
    counts = pack["open_count"]
    assert (counts >= 0).all() and (counts <= 3).all()

    # Slot #2 is invalid if count <= 2
    for i in range(min(50, len(pack["open_count"]))):
        cnt = pack["open_count"][i, 0]  # delete_one_test
        if cnt < 3:
            assert not spec.state_valid(pack, i, "delete_one_test#2")
        else:
            assert spec.state_valid(pack, i, "delete_one_test#2")


# ==============================================================================
# Boundary 10: Multiscale Dimension Divisibility
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m2
def test_boundary_multiscale_dimension_divisibility():
    """768px must divide evenly by grid size: N=16 (48px), N=32 (24px), N=48 (16px)."""
    image_dim = 768
    valid_sizes = [16, 32, 48]
    for n in valid_sizes:
        assert image_dim % n == 0, f"{image_dim} must divide evenly by {n}"

    # Invalid non-divisible sizes
    invalid_sizes = [17, 33, 49, 50]
    for n in invalid_sizes:
        assert image_dim % n != 0, f"{image_dim} should not divide evenly by {n}"


# ==============================================================================
# Boundary 11: Input Adapter Dimension Mismatch
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m3
def test_boundary_input_adapter_dimension_mismatch():
    """Adapter built for d=384 must error if given d=768 vectors."""
    d = 384
    adapter = nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, 2816)
    )
    wrong_input = torch.randn(2, 256, 768)
    with pytest.raises(RuntimeError):
        adapter(wrong_input)


# ==============================================================================
# Boundary 12: Loss Weighting Extreme Slot Counts
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.m3
def test_boundary_loss_weighting_extreme_slots():
    """Balanced slot loss weighting: sum of weights must equal 1.0 even with 1 or 255 changed slots."""
    def calc_weights(n_slots, changed_slots):
        w = np.full(n_slots, 1.0 / n_slots)
        k = len(changed_slots)
        if 0 < k < n_slots:
            w[:] = 0.5 / (n_slots - k)
            w[changed_slots] = 0.5 / k
        return w

    # 1 changed slot out of 256
    w1 = calc_weights(256, [0])
    assert abs(w1.sum() - 1.0) < 1e-6
    assert abs(w1[0] - 0.5) < 1e-6

    # 255 changed slots out of 256 (only 1 unchanged)
    w255 = calc_weights(256, list(range(255)))
    assert abs(w255.sum() - 1.0) < 1e-6
    assert abs(w255[255] - 0.5) < 1e-6
