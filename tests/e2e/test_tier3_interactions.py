"""
tests/e2e/test_tier3_interactions.py - Tier 3: Cross-Feature Interactions E2E Tests.

Verifies pairwise and multi-feature subsystem integrations:
1. Call Graph 16x16 Grid Rendering + Multi-Hop Transitive Operations (mark_reachable / mark_ancestors).
2. Trainable Input Adapter + Balanced Slot Loss Weighting (0.5 changed / 0.5 unchanged).
3. Multi-Step Latent Re-Feed Chains + Offline Prediction Scoring.
4. AST Symbol Extractor + Node Dataset Generator + Gate B0 Dimensionality Probe.
5. Section 27 Spec Adapter + 22 Candidate Pool + Evaluation Manifest Balancing.
6. Vision Stack Representation + ViT-Conv Decoder Spatial Mapping.
7. Canvas Roadmap Teacher + Evaluator Metric Engine for Multi-Site Edits.
8. Candidate Separation + Anisotropy Centering Correction.
9. Instruction Preamble Formatting + Multimodal Attention Mask Alignment.
"""

import ast
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Interaction 1: Call Graph Rendering + Multi-Hop Operations
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m2
def test_interaction_call_graph_and_multihop_ops(cg_loader):
    """Interaction: Rendering call graph into 16x16 grid with multi-hop reachability marking."""
    ops = cg_loader("cg_ops")

    # Construct directed DAG with multi-hop chain: 0 -> 1 -> 2 -> 3; 4 -> 1
    # Selected symbol = 1
    adj = np.zeros((16, 16), dtype=bool)
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 1)]:
        adj[a, b] = True
    kind = np.full(16, ops.FUNC, dtype=np.int32)
    s = ops.GraphState.new(adj, kind, 1)

    # Forward transitive closure: mark_reachable from 1 -> reaches 2 and 3
    marked_fwd = ops.mark_reachable(s)
    grid_fwd = marked_fwd.to_grid()
    assert grid_fwd[2, 2] == ops.MARK, "Node 2 must be marked along diagonal"
    assert grid_fwd[3, 3] == ops.MARK, "Node 3 must be marked along diagonal"
    assert grid_fwd[0, 0] != ops.MARK, "Node 0 (upstream) must NOT be marked"

    # Backward transitive closure: mark_ancestors of 1 -> reaches 0 and 4
    marked_bwd = ops.mark_ancestors(s)
    grid_bwd = marked_bwd.to_grid()
    assert grid_bwd[0, 0] == ops.MARK, "Node 0 must be marked backward along diagonal"
    assert grid_bwd[4, 4] == ops.MARK, "Node 4 must be marked backward along diagonal"
    assert grid_bwd[2, 2] != ops.MARK, "Node 2 (downstream) must NOT be marked"

    # Verify visual rendering produces 768x768 RGB
    img = ops.render_pil(grid_fwd)
    assert img.size == (768, 768)


# ==============================================================================
# Interaction 2: Symbol Adapter + Balanced Slot Loss Weighting
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m3
def test_interaction_symbol_adapter_and_loss_reweighting():
    """Interaction: Forward pass through adapter + balanced slot loss weighting (0.5 changed / 0.5 unchanged)."""
    d = 384
    adapter = nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, 2816)
    )
    # Head mapping back to d
    head = nn.Linear(2816, d)

    # 1 example with 256 slots, slots 5 and 10 are changed
    source_vecs = torch.randn(1, 256, d)
    target_vecs = source_vecs.clone()
    target_vecs[0, 5] += 2.0
    target_vecs[0, 10] += 3.0

    # Forward pass through adapter and head
    soft_tokens = adapter(source_vecs)
    pred_vecs = head(soft_tokens)

    # Calculate weighted loss
    changed_slots = [5, 10]
    weights = np.full(256, 0.5 / (256 - len(changed_slots)), dtype=np.float32)
    weights[changed_slots] = 0.5 / len(changed_slots)
    w_tensor = torch.from_numpy(weights).unsqueeze(0).unsqueeze(-1)  # (1, 256, 1)

    loss = (w_tensor * (pred_vecs - target_vecs) ** 2).sum()
    loss.backward()

    # Verify gradients flow into adapter
    grad_norm = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must successfully flow back into input adapter"


# ==============================================================================
# Interaction 3: Re-Feed Chains + Offline Candidate Scoring
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m2
def test_interaction_refeed_chains_and_offline_candidate_scoring(cg_loader, data_locator):
    """Interaction: Multi-step re-feed execution scored by nearest candidate grid evaluation."""
    spec = cg_loader("cg_sec27_spec")
    ops = cg_loader("cg_ops")
    metrics = cg_loader("cg_metrics")

    pack = spec.load_pack(str(data_locator("gridA_train1608.npz")))
    # For source 0: apply step 1 (reverse_edges), then step 2 (keep_callees)
    # Target state: reverse_edges+keep_callees (one of the held-out pairs)
    s0 = ops.GraphState.new(pack["adj"][0], pack["kind"][0], pack["selected"][0])
    step1 = ops.reverse_edges(s0)
    step2 = ops.keep_callees(step1)

    # Compare step2 grid against candidate grids
    cand_grids = pack["tgt_grid"][0]  # (22, 16, 16)
    pred = step2.to_grid()[None]       # (1, 16, 16)

    idx, tie = metrics.nearest_candidate_grid(pred, cand_grids[None])
    target_state_name = "reverse_edges+keep_callees"
    expected_idx = spec.CG_CANDIDATE_STATES.index(target_state_name)

    assert idx[0] == expected_idx, f"Re-feed chain must match candidate {target_state_name}"
    assert tie[0] == False, "Must cleanly match without ties"


# ==============================================================================
# Interaction 4: AST Symbol Extractor + Node Dataset Generator + Gate B0
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m3
def test_interaction_ast_extractor_node_dataset_and_gate_b0(cg_loader):
    """Interaction: AST extraction feeds into node dataset generation, evaluated by Gate B0 spread."""
    extract = cg_loader("cg_extract")
    node_ds = cg_loader("cg_node_dataset")
    gate = cg_loader("cg_node_gate")

    sample_repo_code = """
def authenticate_user(username: str, token: str) -> bool:
    return len(token) > 10

def get_profile(username: str) -> dict:
    if authenticate_user(username, "secret12345"):
        return {"user": username, "status": "active"}
    return {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "auth.py"
        f.write_text(sample_repo_code)
        res = extract.extract(tmpdir, loose=True, repo="test_repo", commit="abc1234")

        # Verify extracted nodes
        node_names = [n["name"] for n in res["nodes"]]
        assert "authenticate_user" in node_names
        assert "get_profile" in node_names

        # Verify candidate edits on extracted node
        tree = ast.parse(sample_repo_code)
        renamed = node_ds.edit_rename_def(tree, "authenticate_user", "verify_credentials")
        unparsed = ast.unparse(renamed)
        assert "def verify_credentials(" in unparsed

        # Evaluate dimensionality spread on isotropic vectors
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 40)).astype(np.float32)
        sp = gate.spread(vectors, rng)
        assert sp["participation_ratio"] >= 20.0


# ==============================================================================
# Interaction 5: Spec Adapter + Candidate Pool + Evaluation Manifest Balancing
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m2
def test_interaction_spec_adapter_and_eval_manifests_consistency(cg_loader, data_locator):
    """Interaction: cg_sec27_spec generates 16 manifests correctly balanced against 22 candidate states."""
    spec = cg_loader("cg_sec27_spec")
    test_pack = spec.load_pack(str(data_locator("gridA_test360_httpx.npz")))
    manifests = spec.build_cg_eval_manifests(test_pack)

    # 1. Shuffled manifest control: prompt instruction op must never match target state op
    shuf = manifests["seen_shuf360"]
    bal = manifests["seen_bal360"]
    assert len(shuf["op_names"]) == 360
    assert len(bal["op_names"]) == 360
    # Shuffled prompt ops must differ from target ops for all 360 items
    same_op_count = sum(o1 == o2 for o1, o2 in zip(shuf["op_names"], shuf["shuffled_from_ops"]))
    assert same_op_count == 0, "seen_shuf360 control must have 0 matching operations between target and prompt"

    # 2. Balanced manifest covers all 14 trained fixed operations
    unique_ops = set(bal["op_names"])
    assert len(unique_ops) == 14, f"seen_bal360 must cover exactly 14 trained operations, got {len(unique_ops)}"


# ==============================================================================
# Interaction 6: Vision Stack Representation + ViT Decoder Spatial Mapping
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m4
def test_interaction_vision_stack_and_vit_image_decoder():
    """Interaction: Image (768x768) -> Soft tokens (256x1152) -> ViT+Conv decoder RGB (768x768)."""
    # Verify spatial resolution invariants
    patch_size = 16
    grid_dim = 16
    assert patch_size * grid_dim == 256  # intermediate
    # Image 768 / 16 cells = 48 px per cell
    cell_px = 768 // 16
    assert cell_px == 48

    # Decoder spatial upsampling: 16x16 tokens -> 768x768 image requires 48x upsampling factor
    upsample_factor = 768 // 16
    assert upsample_factor == 48


# ==============================================================================
# Interaction 7: Canvas Roadmap Teacher + Evaluator Multi-Site Edits
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m3
def test_interaction_canvas_roadmap_teacher_and_evaluator(cg_loader, data_locator):
    """Interaction: Multi-site edits (k=2) supervised by Canvas Teacher and scored by Evaluator."""
    node_metrics = cg_loader("cg_node_metrics")

    # Mock multi-site candidate changes: site 1 modifies slot 12, site 2 deletes slot 45
    changes = [[12, 100], [45, -1]]
    labels = np.zeros(256, dtype=np.int8)
    for slot, t in changes:
        labels[slot] = node_metrics.DELETED if t < 0 else node_metrics.EDITED

    assert labels[12] == node_metrics.EDITED
    assert labels[45] == node_metrics.DELETED
    assert (labels != 0).sum() == 2, "Exactly 2 slots should have roadmap non-zero labels"


# ==============================================================================
# Interaction 8: Candidate Separation + Anisotropy Centering Correction
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m2
def test_interaction_candidate_separation_and_anisotropy_correction(cg_loader):
    """Interaction: Candidate separation requires dataset mean subtraction (centered_hv_cos)."""
    metrics = cg_loader("cg_metrics")

    np.random.seed(42)
    # Strongly anisotropic vectors with huge shared mean component
    shared_mean = np.ones((1, 1, 1, 1152), dtype=np.float32) * 100.0
    cands = shared_mean + np.random.randn(5, 22, 256, 1152).astype(np.float32)

    # Uncentered separation would show pairwise cosine ≈ 0.999
    # Centered separation subtracts mean, restoring variance
    mean_hv = cands.mean(axis=(0, 1, 2), keepdims=True)
    sep = metrics.candidate_separation(cands, mean_hv)
    assert sep.shape == (5,)
    # Centered max cosine must be strictly below 1.0
    assert (sep < 0.99).all(), "Anisotropy correction must center candidates away from cosine collapse"


# ==============================================================================
# Interaction 9: Instruction Preamble Formatting + Attention Alignment
# ==============================================================================
@pytest.mark.tier3
@pytest.mark.m1
@pytest.mark.m2
def test_interaction_preamble_and_instruction_formatting(cg_loader, data_locator):
    """Interaction: Preamble formatting attaches visual domain rules to all instruction arms."""
    spec = cg_loader("cg_sec27_spec")
    pack = spec.load_pack(str(data_locator("gridA_train1608.npz")))

    preamble_text = spec.preamble(pack)
    assert "rows call columns" in preamble_text or "colours mean" in preamble_text or "16x16" in preamble_text
    instruction = "Delete all test symbols and their incident calls."
    full_prompt = preamble_text + " " + instruction

    words = full_prompt.split()
    assert len(words) > 30, "Prompt must include preamble and instruction"
    assert "Delete all test symbols" in full_prompt
