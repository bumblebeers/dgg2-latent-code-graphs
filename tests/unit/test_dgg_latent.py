#!/usr/bin/env python3
"""Unit tests for dgg_latent library."""

import math
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgg_latent import (
    ContinuousLatentSampler,
    CrossSoftCanvasHead,
    DualReadoutSoftCanvasHead,
    ExpBInputAdapter,
    PhaseController,
    SoftCanvasHead,
    SymbolVectorInputAdapter,
    _GenRecorder,
    balanced_slot_loss,
    build_single_canvas_head,
    compute_readout_input,
    corrupt_discrete_canvas,
    cosine_corruption_schedule,
    distribution_ce,
    install_dg_attn_fix,
    linear_corruption_schedule,
    load_stock_generation_config,
    loss_hv,
    loss_pixel,
    per_example_reliability_weight,
    set_mask_variant,
    slerp,
    stock_generate,
    stock_generate_with_readout,
)


def test_soft_canvas_head_shapes_and_init():
    head_1152 = SoftCanvasHead(d_in=2816, d_bot=768, d_out=1152)
    assert head_1152.num_params() > 3_000_000
    head_1152.init_cold_start()

    x = torch.randn(2, 256, 2816)
    out = head_1152(x)
    assert out.shape == (2, 256, 1152)
    assert out.dtype == torch.float32

    # Experiment B configuration (d_out=384)
    head_384 = SoftCanvasHead(d_in=2816, d_bot=768, d_out=384)
    head_384.init_cold_start()
    out_384 = head_384(x)
    assert out_384.shape == (2, 256, 384)


def test_dual_readout_soft_canvas_head():
    head = DualReadoutSoftCanvasHead(d_in=2816, d_bot=768, d_out=1152)
    head.init_cold_start()
    assert head.num_params() > 6_000_000

    b, l = 2, 256
    h_dec = torch.randn(b, l, 2816, requires_grad=True)
    e_soft = torch.randn(b, l, 2816, requires_grad=True)
    c_z = torch.rand(b, l, 2)

    # Forward via separate arguments
    out = head(h_dec, e_soft=e_soft, c_z=c_z)
    assert out.shape == (b, l, 1152)
    assert 0.0 <= head.last_gate_mean <= 1.0

    # Gradient flow test
    loss = out.sum()
    loss.backward()
    assert h_dec.grad is not None and not torch.isnan(h_dec.grad).any()
    assert e_soft.grad is not None and not torch.isnan(e_soft.grad).any()

    # Forward via dict pack
    pack = {"h_dec": torch.randn(b, l, 2816), "e_soft": torch.randn(b, l, 2816), "c_z": c_z}
    out_pack = head(pack)
    assert out_pack.shape == (b, l, 1152)

    # Forward via tuple
    out_tuple = head((torch.randn(b, l, 2816), torch.randn(b, l, 2816), c_z))
    assert out_tuple.shape == (b, l, 1152)


def test_cross_soft_canvas_head():
    head = CrossSoftCanvasHead(d_in=2816, d_hidden=768, d_out=1152, heads=8)
    head.init_cold_start()
    assert head.num_params() > 8_000_000

    c1 = torch.randn(2, 256, 2816)
    c2 = torch.randn(2, 256, 2816)
    out = head(c1, c2)
    assert out.shape == (2, 256, 1152)


def test_compute_readout_input():
    b, l, d, v = 2, 64, 128, 500
    h_dec = torch.randn(b, l, d)
    logits = torch.randn(b, l, v)
    lm_weight = torch.randn(v, d)

    # Mode: hidden
    ro_hidden = compute_readout_input(h_dec=h_dec, logits=logits, readout_mode="hidden")
    assert torch.equal(ro_hidden, h_dec)

    # Mode: logits
    ro_logits = compute_readout_input(
        h_dec=h_dec, logits=logits, readout_mode="logits", lm_weight=lm_weight, tau=1.0
    )
    assert ro_logits.shape == (b, l, d)

    # Mode: dual_gated
    ro_dual = compute_readout_input(
        h_dec=h_dec, logits=logits, readout_mode="dual_gated", lm_weight=lm_weight, tau=1.0
    )
    assert isinstance(ro_dual, dict)
    assert "h_dec" in ro_dual and "e_soft" in ro_dual and "c_z" in ro_dual
    assert ro_dual["e_soft"].shape == (b, l, d)
    assert ro_dual["c_z"].shape == (b, l, 2)


def test_build_single_canvas_head_factory():
    head1 = build_single_canvas_head("hidden", d_in=2816, d_out=1152)
    assert isinstance(head1, SoftCanvasHead)

    head2 = build_single_canvas_head("dual_gated", d_in=2816, d_out=1152)
    assert isinstance(head2, DualReadoutSoftCanvasHead)


def test_distribution_ce_crash_guard():
    b, l, v, k = 2, 4, 100, 4
    logits = torch.randn(b, l, v)
    mode_ids = torch.randint(0, v, (b, l, k))
    # Mode 0 has small weight, modes 1..3 have even smaller
    mode_weights = torch.tensor([[[0.1, 0.01, 0.005, 0.001]]]).repeat(b, l, 1)

    # w_thresh=0.5 would prune all modes if not for rank-1 crash guard!
    loss = distribution_ce(logits, mode_ids, mode_weights, w_thresh=0.5, reduction="mean")
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    # Reduction none
    loss_none = distribution_ce(logits, mode_ids, mode_weights, reduction="none")
    assert loss_none.shape == (b, l)


def test_loss_hv_and_pixel():
    pred_hv = torch.randn(2, 256, 1152)
    tgt_hv = torch.randn(2, 256, 1152)
    loss = loss_hv(pred_hv, tgt_hv, cos_weight=2.0)
    assert loss.item() > 0.0

    # Pixel loss unclamped gradient verification
    pred_rgb = torch.tensor([[[[1.5, -0.5], [2.0, -1.0]]]] * 3).unsqueeze(0).detach().requires_grad_(True)
    tgt_rgb = torch.zeros_like(pred_rgb)
    pix_loss = loss_pixel(pred_rgb, tgt_rgb)
    pix_loss.backward()
    # Gradients must be non-zero on out-of-range values (unclamped invariant)
    assert (pred_rgb.grad.abs() > 0.0).all()


def test_per_example_reliability_weight():
    t = torch.tensor([0.2, 0.8, 1.2, -0.1])
    w = per_example_reliability_weight(t, canvas_mode="single")
    expected = torch.tensor([0.8, 0.2, 0.0, 1.0])
    assert torch.allclose(w, expected, atol=1e-5)

    t2 = torch.tensor([0.5, 0.5, 0.0, 0.0])
    w_two = per_example_reliability_weight(t, canvas_mode="two", t2_vec=t2)
    expected_two = expected * torch.tensor([0.5, 0.5, 1.0, 1.0])
    assert torch.allclose(w_two, expected_two, atol=1e-5)


def test_balanced_slot_loss():
    b, l, d = 2, 10, 32
    pred = torch.zeros(b, l, d)
    target = torch.ones(b, l, d)
    mask = torch.zeros(b, l, dtype=torch.bool)
    mask[:, 0] = True  # Only 1 slot out of 10 changed

    loss = balanced_slot_loss(pred, target, mask, weight_changed=0.5, weight_unchanged=0.5)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_phase_controller():
    controller = PhaseController(condition="ce_to_readout", total_steps=1000, phase1_fraction=0.5)
    w_start = controller.get_weights(0)
    assert w_start["ce"] == 1.0 and w_start["hv"] == 0.05

    w_mid = controller.get_weights(500)
    assert w_mid["ce"] == 1.0

    w_end = controller.get_weights(1000)
    assert math.isclose(w_end["ce"], 0.1, abs_tol=1e-3)
    assert math.isclose(w_end["hv"], 1.0, abs_tol=1e-3)


def test_sampling_and_slerp():
    x0 = torch.zeros(4, 256, dtype=torch.long)
    t_zero = torch.zeros(4)
    corrupted_0 = corrupt_discrete_canvas(x0, t_zero, vocab_size=1000)
    assert torch.equal(corrupted_0, x0)

    t_one = torch.ones(4)
    corrupted_1 = corrupt_discrete_canvas(x0, t_one, vocab_size=1000)
    assert not torch.equal(corrupted_1, x0)

    # Slerp
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0, 0.0]])
    mid = slerp(0.5, v1, v2)
    assert torch.allclose(torch.norm(mid, dim=-1), torch.tensor([1.0]), atol=1e-4)

    # Schedules
    assert len(linear_corruption_schedule(10)) == 10
    assert len(cosine_corruption_schedule(10)) == 10


def test_symbol_vector_input_adapter():
    adapter = SymbolVectorInputAdapter(in_dim=384, out_dim=2816, rms_scale=1.28)
    assert isinstance(adapter, ExpBInputAdapter)

    b, l = 2, 256
    x = torch.randn(b, l, 384)
    out = adapter(x)
    assert out.shape == (b, l, 2816)

    # Check RMS scale normalization: rms along last dimension should be ~1.28
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 1.28), atol=1e-4)

    # Test checkpoint save and from_checkpoint strict=True roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "adapter.pt")
        torch.save({"adapter_state_dict": adapter.state_dict()}, ckpt_path)

        loaded = SymbolVectorInputAdapter.from_checkpoint(ckpt_path, device="cpu", strict=True)
        assert loaded.in_dim == 384
        assert loaded.out_dim == 2816
        out_loaded = loaded(x)
        assert torch.allclose(out, out_loaded, atol=1e-6)


def test_attn_fix_api():
    # Test setting mask variant
    set_mask_variant("gemma4")
    set_mask_variant("dg_generate")
    set_mask_variant("causal")
    with pytest.raises(ValueError):
        set_mask_variant("invalid_variant")
    set_mask_variant("gemma4")


def test_load_stock_generation_config():
    dg_path = os.path.expanduser("~/ml-models/diffusiongemma-26B-A4B-it")
    if not os.path.exists(dg_path):
        pytest.skip("DG model dir not found")

    gc, prov = load_stock_generation_config(dg_path)
    assert prov["sampler"] == "EntropyBoundSampler"
    assert prov["stopping"] == "StableAndConfidentStoppingCriteria"
    assert prov["entropy_bound"] == 0.1
    assert prov["max_denoising_steps"] == 48


def test_stock_generate_and_with_readout_mock():
    class MockNorm(nn.Module):
        def forward(self, x):
            return x

    class MockDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = MockNorm()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = MockDecoder()

        def _denoising_step(self, decoder_forward, *args, **kwargs):
            # Simulate denoising step returning (sample, argmax_canvas, processed_logits)
            df_out = decoder_forward()
            canvas = torch.randint(0, 1000, (1, 256))
            proc_logits = torch.randn(1, 256, 1000)
            return (canvas, canvas, proc_logits)

        def _prepare_diffusion_stopping_criteria(self, generation_config):
            def criteria(canvas, logits):
                return torch.tensor([True])
            return criteria

        def generate(self, **kwargs):
            b = 1
            max_new = kwargs.get("max_new_tokens", 256)
            prompt_len = kwargs.get("input_ids").shape[1]
            seq_len = prompt_len + max_new

            def dummy_df():
                self.decoder.norm(torch.randn(b, max_new, 2816))
                class DFOut:
                    logits = torch.randn(b, max_new, 1000)
                return DFOut()

            self._denoising_step(dummy_df)

            class Out:
                sequences = torch.zeros(b, seq_len, dtype=torch.long)
                tokens_per_forward = torch.tensor([256.0])
            return Out()

    model = MockModel()
    inputs = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}

    # Test stock_generate
    res = stock_generate(model, inputs, max_new_tokens=256)
    assert res["sequences"].shape == (1, 266)
    assert res["canvas"].shape == (1, 256)
    assert res["hidden"].shape == (1, 256, 2816)

    # Test stock_generate_with_readout
    head = SoftCanvasHead(d_in=2816, d_bot=256, d_out=1152)
    res_ro = stock_generate_with_readout(
        base_model=model,
        inputs=inputs,
        heads=head,
        readout_mode="hidden",
        max_new_tokens=256,
    )
    assert "hv_res" in res_ro
    assert res_ro["hv_res"].shape == (1, 256, 1152)
    assert res_ro["readout_source"] == "final_sc_denoising_step"
