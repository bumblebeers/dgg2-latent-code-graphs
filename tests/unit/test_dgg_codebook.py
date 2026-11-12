#!/usr/bin/env python3
"""Unit tests for dgg_codebook library."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from dgg_codebook import (
    BackgroundSuppression,
    ContextualizedSpatialMLP,
    CtxSpatialTeacher,
    DistilledSpatialTeacher,
    FrequencyDebiaser,
    IsoSpatialTeacher,
    RoPECarrierDebiaser,
    SurrogateDistillationLoss,
    apply_foreground_salience_boost,
    build_multimodal_target_distribution,
    classify_lexical_token_class,
    compute_expected_vocabulary_embedding,
    extract_teacher_modes_from_hv_batched,
    sample_x0_from_distribution,
)


def test_build_multimodal_target_distribution():
    b, l, k = 2, 4, 8
    mode_ids = torch.randint(0, 1000, (b, l, k))
    # Rank 0 has small weight, modes 1..7 have decreasing weights
    weights = torch.tensor([[[0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0, 0.0]]]).repeat(b, l, 1)

    ids_out, q_tilde = build_multimodal_target_distribution(mode_ids, weights, w_thresh=0.2)
    assert torch.equal(ids_out, mode_ids)
    assert torch.allclose(q_tilde.sum(dim=-1), torch.ones(b, l), atol=1e-5)
    # Crash guard ensures rank 0 is preserved
    assert (q_tilde[..., 0] > 0.0).all()


def test_compute_expected_vocabulary_embedding():
    b, l, k, d = 2, 4, 3, 16
    q_tilde = torch.zeros(b, l, k)
    q_tilde[..., 0] = 1.0  # 100% on mode 0

    mode_ids = torch.zeros(b, l, k, dtype=torch.long)
    mode_ids[..., 0] = 5
    mode_ids[..., 1] = 10
    mode_ids[..., 2] = 20

    vocab_emb = nn.Embedding(50, d)
    e_soft = compute_expected_vocabulary_embedding(q_tilde, mode_ids, vocab_emb)
    expected_mode0 = vocab_emb.weight[5].unsqueeze(0).unsqueeze(0).expand(b, l, d)
    assert torch.allclose(e_soft, expected_mode0, atol=1e-5)

    # 50/50 superposition test
    q_tilde[..., 0] = 0.5
    q_tilde[..., 1] = 0.5
    e_soft_half = compute_expected_vocabulary_embedding(q_tilde, mode_ids, vocab_emb)
    expected_half = 0.5 * (vocab_emb.weight[5] + vocab_emb.weight[10]).unsqueeze(0).unsqueeze(0).expand(b, l, d)
    assert torch.allclose(e_soft_half, expected_half, atol=1e-5)


def test_apply_foreground_salience_boost():
    weights = torch.ones(2, 4, 5)
    is_fg = torch.tensor([[True, False, True, False], [False, True, False, True]])

    boosted = apply_foreground_salience_boost(weights, is_fg, boost_factor=2.0)
    # Mode 0 is untouched, modes 1..4 on FG positions are doubled
    assert (boosted[is_fg, 1:] == 2.0).all()
    assert (boosted[~is_fg, :] == 1.0).all()
    assert (boosted[is_fg, 0] == 1.0).all()


def test_sample_x0_from_distribution():
    b, l, k = 2, 8, 4
    mode_ids = torch.tensor([[[10, 20, 30, 40]]]).repeat(b, l, 1)
    q_tilde = torch.zeros(b, l, k)
    q_tilde[..., 1] = 1.0  # All mass on token 20

    sampled = sample_x0_from_distribution(mode_ids, q_tilde)
    assert sampled.shape == (b, l)
    assert (sampled == 20).all()


def test_frequency_debiaser():
    debiaser = FrequencyDebiaser(vocab_size=100, gamma=0.5)
    # Simulate token 1 being very frequent and token 2 being rare
    tokens = torch.tensor([1] * 1000 + [2] * 10)
    debiaser.update(tokens)

    mode_ids = torch.tensor([[1, 2]])
    mults = debiaser.get_multipliers(mode_ids)
    # Frequent token multiplier must be lower than rare token multiplier
    assert mults[0, 0] < mults[0, 1]


def test_background_suppression():
    suppressor = BackgroundSuppression(bg_token_id=0, penalty=0.25)
    mode_ids = torch.tensor([[[0, 1], [0, 1]]])
    weights = torch.ones(1, 2, 2)
    spatial_var = torch.tensor([[0.5, 0.01]])  # Pos 0 is active, pos 1 is flat

    adj = suppressor.suppress(mode_ids, weights, spatial_var, var_threshold=0.1)
    # Pos 0 mode 0 (bg on active patch) should be suppressed
    assert adj[0, 0, 0] == 0.25
    # Pos 1 mode 0 (bg on flat patch) should not be suppressed
    assert adj[0, 1, 0] == 1.0


def test_rope_carrier_debiaser():
    canvas_len, d = 256, 64
    debiaser = RoPECarrierDebiaser(canvas_len=canvas_len, embed_dim=d)

    # Synthetic carrier + noise
    true_carrier = torch.randn(canvas_len, d)
    for _ in range(5):
        batch = true_carrier.unsqueeze(0).repeat(4, 1, 1) + torch.randn(4, canvas_len, d) * 0.1
        debiaser.fit_batch(batch)

    assert torch.allclose(debiaser.carrier, true_carrier, atol=0.2)

    # Subtraction
    sample = true_carrier.unsqueeze(0).repeat(2, 1, 1)
    sub = debiaser.subtract(sample)
    assert torch.norm(sub) < torch.norm(sample)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "carrier.pt")
        debiaser.save(path)
        debiaser2 = RoPECarrierDebiaser(canvas_len=canvas_len, embed_dim=d)
        debiaser2.load(path)
        assert torch.equal(debiaser.carrier, debiaser2.carrier)


def test_spatial_teachers_and_loss():
    b, l, d_in = 2, 16, 1152
    h_v = torch.randn(b, l, d_in)

    # IsoSpatialTeacher
    iso = IsoSpatialTeacher(d_in=d_in, d_hidden=128, num_modes=8, d_emb=64)
    mode_vecs_iso, sal_iso = iso(h_v)
    assert mode_vecs_iso.shape == (b, l, 8, 64)
    assert sal_iso.shape == (b, l, 8)
    assert (sal_iso >= 0.0).all() and (sal_iso <= 1.0).all()

    # CtxSpatialTeacher
    ctx = CtxSpatialTeacher(d_in=d_in, d_hidden=128, num_modes=8, d_emb=64, n_layers=2, n_heads=4, num_patches=16)
    mode_vecs_ctx, sal_ctx = ctx(h_v)
    assert mode_vecs_ctx.shape == (b, l, 8, 64)
    assert sal_ctx.shape == (b, l, 8)

    # Distillation loss
    loss_fn = SurrogateDistillationLoss(weight_salience=1.0, weight_vector=2.0)
    losses = loss_fn(mode_vecs_iso, sal_iso, mode_vecs_ctx.detach(), sal_ctx.detach())
    assert "loss_total" in losses
    assert losses["loss_total"].item() > 0.0


def test_distilled_spatial_teacher_real_checkpoints():
    dist_dir = os.path.expanduser("~/workspaces/arc-agi-3/rendered/dgg2/distillation")
    ctx_path = os.path.join(dist_dir, "model_ctx_clean_256.pt")
    iso_path = os.path.join(dist_dir, "model_iso_clean_256.pt")

    if not os.path.exists(ctx_path) or not os.path.exists(iso_path):
        pytest.skip("Real distillation checkpoints not found in arc-agi-3")

    # 1. Test model_ctx_clean_256.pt loading with strict=True
    m_ctx = DistilledSpatialTeacher.from_checkpoint(ctx_path, device="cpu", strict=True)
    assert isinstance(m_ctx, ContextualizedSpatialMLP)
    assert m_ctx.in_dim == 1152
    assert m_ctx.vocab_size == 11964
    x_ctx = torch.randn(2, 256, 1152)
    with torch.no_grad():
        out_ctx = m_ctx(x_ctx)
    assert "logits_pred" in out_ctx
    assert out_ctx["logits_pred"].shape == (2, 256, 11964)

    # 2. Test model_iso_clean_256.pt loading with strict=True
    m_iso = DistilledSpatialTeacher.from_checkpoint(iso_path, device="cpu", strict=True)
    assert isinstance(m_iso, ContextualizedSpatialMLP)
    assert m_iso.in_dim == 1233
    assert m_iso.vocab_size == 11964
    x_iso = torch.randn(2, 256, 1233)
    with torch.no_grad():
        out_iso = m_iso(x_iso)
    assert "logits_pred" in out_iso
    assert out_iso["logits_pred"].shape == (2, 256, 11964)


def test_classify_lexical_token_class():
    assert classify_lexical_token_class("hello") == "english_word"
    assert classify_lexical_token_class("Python") == "english_word"
    assert classify_lexical_token_class("🚀") == "emoji_icon"
    assert classify_lexical_token_class("❤️") == "emoji_icon"
    assert classify_lexical_token_class("┌") == "unicode_glyph"
    assert classify_lexical_token_class("→") == "unicode_glyph"
    assert classify_lexical_token_class("你好") == "cjk_multilingual"
    assert classify_lexical_token_class("한글") == "cjk_multilingual"
    assert classify_lexical_token_class("") == "other"


def test_extract_teacher_modes_batched_mock():
    # Synthetic mock checkpoint matching ctx structure
    k_cand = 50
    state_dict = {
        "net.0.weight": torch.randn(2048, 1152),
        "net.0.bias": torch.randn(2048),
        "net.2.weight": torch.ones(2048),
        "net.2.bias": torch.zeros(2048),
        "net.3.weight": torch.randn(2048, 2048),
        "net.3.bias": torch.randn(2048),
        "net.5.weight": torch.ones(2048),
        "net.5.bias": torch.zeros(2048),
        "net.6.weight": torch.randn(k_cand, 2048),
        "net.6.bias": torch.randn(k_cand),
    }
    cand_vocab = torch.arange(k_cand, dtype=torch.int32)
    mock_ckpt = {
        "state_dict": state_dict,
        "k_cand": k_cand,
        "cand_vocab": cand_vocab,
        "stats": {
            "mu": torch.zeros(1152),
            "std": torch.ones(1152),
        },
        "penalty_sm": torch.ones(k_cand),
    }

    class MockTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {0: "word", 1: "🚀", 2: "→", 3: "汉"}
            return mapping.get(ids[0] % 4, "other")

    hv_all = torch.randn(2, 256, 1152)
    ids, w = extract_teacher_modes_from_hv_batched(
        hv_all=hv_all,
        ctx_ckpt=mock_ckpt,
        tokenizer=MockTokenizer(),
        device="cpu",
        batch_size=1,
    )
    assert ids.shape[0] == 2
    assert ids.shape[1] == 256
    assert ids.shape[2] <= 64
    assert w.shape == ids.shape
    assert (w >= 0.0).all()
