#!/usr/bin/env python3
"""Unit tests for dgg_vit_decoder library."""

import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from dgg_vit_decoder import (
    ConvDecoder,
    ImageDecoder,
    PreLNBlock,
    ResBlock,
    TokenMixer,
    UpsampleTo,
    VisionStack,
    pred_to_grids,
    psnr,
    ssim,
)


def test_token_mixer_shapes():
    b, n, d_in, d = 2, 256, 1152, 768
    mixer = TokenMixer(d_in=d_in, d=d, depth=2, heads=8, mlp=1024, grid=16)
    h_v = torch.randn(b, n, d_in)
    out = mixer(h_v)
    assert out.shape == (b, d, 16, 16)


def test_resblock_and_upsample_to():
    x = torch.randn(2, 64, 16, 16)
    res = ResBlock(64)
    out_res = res(x)
    assert out_res.shape == (2, 64, 16, 16)

    # PixelShuffle scale 3
    up3 = UpsampleTo(c_in=64, c_out=32, scale=3)
    out_up3 = up3(x)
    assert out_up3.shape == (2, 32, 48, 48)

    # PixelShuffle scale 2
    up2 = UpsampleTo(c_in=32, c_out=16, scale=2)
    out_up2 = up2(out_up3)
    assert out_up2.shape == (2, 16, 96, 96)


def test_conv_decoder_mid_grey_init():
    decoder = ConvDecoder(c_in=768)
    # Output conv should be zero-initialized
    assert torch.all(decoder.out.weight == 0.0)
    assert torch.all(decoder.out.bias == 0.0)

    # Therefore sigmoid(0) = 0.5 (mid-grey output)
    feat = torch.randn(1, 768, 16, 16)
    rgb = decoder(feat)
    assert rgb.shape == (1, 3, 768, 768)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.5), atol=1e-5)


def test_image_decoder_architecture_and_params():
    decoder = ImageDecoder()
    n_params = decoder.num_params()
    # Architecture has ~34.15M parameters
    assert 33_000_000 < n_params < 36_000_000, f"Expected ~34.15M params, got {n_params}"

    h_v = torch.randn(1, 256, 1152)
    rgb = decoder(h_v)
    assert rgb.shape == (1, 3, 768, 768)
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()


def test_image_decoder_save_and_from_pretrained():
    decoder = ImageDecoder()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "decoder.pt")
        torch.save(decoder.state_dict(), ckpt_path)

        loaded = ImageDecoder.from_pretrained(ckpt_path)
        assert loaded.num_params() == decoder.num_params()

        loaded_ckpt = ImageDecoder.from_checkpoint(ckpt_path, device="cpu", strict=True)
        assert loaded_ckpt.num_params() == decoder.num_params()


def test_image_decoder_from_checkpoint_real():
    real_ckpt = os.path.expanduser(
        "~/workspaces/arc-agi-3/rendered/arc3/image_decoder/s2_vitconv/decoder.pt"
    )
    if not os.path.exists(real_ckpt):
        pytest.skip(f"Real checkpoint not present: {real_ckpt}")

    model = ImageDecoder.from_checkpoint(real_ckpt, device="cpu", strict=True)
    assert model.num_params() == 34_149_747
    h_v = torch.randn(1, 256, 1152)
    with torch.no_grad():
        rgb = model(h_v)
    assert rgb.shape == (1, 3, 768, 768)
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()


def test_pred_to_grids():
    # 8-class test palette
    palette = np.array(
        [
            [0, 0, 0],       # 0: black
            [255, 0, 0],     # 1: red
            [0, 255, 0],     # 2: green
            [0, 0, 255],     # 3: blue
            [255, 255, 0],   # 4: yellow
            [255, 0, 255],   # 5: magenta
            [0, 255, 255],   # 6: cyan
            [255, 255, 255], # 7: white
        ],
        dtype=np.uint8,
    )

    # Build 768x768 image with known 16x16 grid pattern
    # Cell size = 768 // 16 = 48 px
    grid_truth = np.random.randint(0, 8, (1, 16, 16), dtype=np.uint8)
    img_np = np.zeros((1, 3, 768, 768), dtype=np.float32)

    for r in range(16):
        for c in range(16):
            cls_id = grid_truth[0, r, c]
            rgb_val = palette[cls_id] / 255.0
            for ch in range(3):
                img_np[0, ch, r * 48 : (r + 1) * 48, c * 48 : (c + 1) * 48] = rgb_val[ch]

    img_tensor = torch.from_numpy(img_np)
    recovered_grid = pred_to_grids(img_tensor, palette, grid_dim=16)
    assert np.array_equal(recovered_grid, grid_truth)


def test_metrics_psnr_and_ssim():
    img1 = torch.rand(2, 3, 64, 64)
    # Identical images
    score_psnr_ident = psnr(img1, img1)
    assert (score_psnr_ident > 80.0).all()

    score_ssim_ident = ssim(img1, img1)
    assert torch.allclose(score_ssim_ident, torch.ones(2), atol=1e-3)

    # Noisy image
    img2 = torch.clamp(img1 + torch.randn_like(img1) * 0.1, 0.0, 1.0)
    score_psnr_noisy = psnr(img1, img2)
    assert (score_psnr_noisy < score_psnr_ident).all()
    assert (score_psnr_noisy > 10.0).all()


def test_vision_stack_interface():
    class MockVisionTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 1152)

        def forward(self, pixel_values, pixel_position_ids=None):
            class Out:
                last_hidden_state = torch.randn(pixel_values.shape[0] * 256, 1152)
            return Out()

    class MockEmbedVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1152, 2816)

        def forward(self, inputs_embeds):
            return self.proj(inputs_embeds)

    v_tower = MockVisionTower()
    e_proj = MockEmbedVision()
    device = torch.device("cpu")
    v_stack = VisionStack(v_tower, e_proj, device=device)

    assert v_stack.num_params() > 0

    pv = torch.randn(2, 2520, 768)
    pos = torch.zeros(2, 2520, 2)
    out = v_stack.encode(pv, pos)
    assert out["h_v"].shape == (2, 256, 1152)
    assert out["softs"].shape == (2, 256, 2816)
