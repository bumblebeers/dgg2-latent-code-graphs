#!/usr/bin/env python3
"""DGG Codebook: Isolated and Contextual Spatial Teacher Models.

Provides fast surrogate distillation models (IsoSpatialTeacher, CtxSpatialTeacher)
that map continuous patch features h_v to high-precision vocabulary modes and salience
weights without running 26B multi-prompt elicitation at inference time.
"""
from __future__ import annotations

import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

HV_DIM = 1152
DEFAULT_K_MODES = 16


class IsoSpatialTeacher(nn.Module):
    """Isolated spatial teacher predicting local photometric/appearance modes from h_v[i].

    Operates pointwise per patch:
        h_{v, i} in R^{1152}
        -> LayerNorm + MLP
        -> mode_logits in R^{K_modes x V} or mode_features in R^{K_modes x d_emb}
        -> salience_weights in R^{K_modes}
    """

    def __init__(
        self,
        d_in: int = HV_DIM,
        d_hidden: int = 768,
        num_modes: int = DEFAULT_K_MODES,
        d_emb: int = 2816,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.num_modes = num_modes
        self.d_emb = d_emb

        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

        self.mode_proj = nn.Linear(d_hidden, num_modes * d_emb)
        self.salience_head = nn.Linear(d_hidden, num_modes)

    def forward(self, h_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict continuous mode vectors and salience weights.

        Args:
            h_v: (B, L, d_in) or (B, d_in) patch latent tensor.

        Returns:
            Tuple of:
                mode_vectors: (B, L, num_modes, d_emb)
                salience_weights: (B, L, num_modes) in [0, 1]
        """
        feats = self.net(h_v.float())
        b = feats.shape[0]
        l = feats.shape[1] if feats.dim() == 3 else 1

        mode_vecs = self.mode_proj(feats).reshape(b, l, self.num_modes, self.d_emb)
        salience = torch.sigmoid(self.salience_head(feats)).reshape(b, l, self.num_modes)

        if feats.dim() == 2:
            return mode_vecs.squeeze(1), salience.squeeze(1)
        return mode_vecs, salience


class CtxSpatialTeacher(nn.Module):
    """Contextual spatial teacher predicting topological/boundary roles across canvas.

    Processes full 256-patch spatial context using multi-head self-attention:
        H_v in R^{B x 256 x 1152}
        -> Positional Embedding + Transformer Blocks
        -> mode_vectors in R^{B x 256 x K x d_emb}
        -> salience_weights in R^{B x 256 x K}
    """

    def __init__(
        self,
        d_in: int = HV_DIM,
        d_hidden: int = 512,
        num_modes: int = DEFAULT_K_MODES,
        d_emb: int = 2816,
        n_layers: int = 4,
        n_heads: int = 8,
        num_patches: int = 256,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.num_modes = num_modes
        self.d_emb = d_emb

        self.input_proj = nn.Linear(d_in, d_hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, d_hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_hidden)

        self.mode_proj = nn.Linear(d_hidden, num_modes * d_emb)
        self.salience_head = nn.Linear(d_hidden, num_modes)

    def forward(self, h_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict contextual mode vectors and salience scores across canvas."""
        b, l, _ = h_v.shape
        x = self.input_proj(h_v.float()) + self.pos_emb[:, :l, :]
        h = self.norm(self.transformer(x))

        mode_vecs = self.mode_proj(h).reshape(b, l, self.num_modes, self.d_emb)
        salience = torch.sigmoid(self.salience_head(h))
        return mode_vecs, salience


class SurrogateDistillationLoss(nn.Module):
    """Multi-task distillation loss for training spatial teachers against 26B elicitation.

    Combines:
    1. Salience weight MSE/BCE.
    2. Continuous mode representation cosine similarity.
    3. Top-k mode ranking consistency.
    """

    def __init__(self, weight_salience: float = 1.0, weight_vector: float = 2.0):
        super().__init__()
        self.weight_salience = weight_salience
        self.weight_vector = weight_vector

    def forward(
        self,
        pred_modes: torch.Tensor,
        pred_salience: torch.Tensor,
        target_modes: torch.Tensor,
        target_salience: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Salience loss
        loss_sal = F.mse_loss(pred_salience.float(), target_salience.float())

        # Vector alignment loss (cosine distance across modes)
        cos_sim = F.cosine_similarity(pred_modes.float(), target_modes.float(), dim=-1)
        loss_vec = (1.0 - cos_sim).mean()

        total = self.weight_salience * loss_sal + self.weight_vector * loss_vec
        return {
            "loss_total": total,
            "loss_salience": loss_sal,
            "loss_vector": loss_vec,
        }


# copied from arc-agi-3@5396d0d:scripts/dgg2/colab_runner_phase2_transforms.py
class ContextualizedSpatialMLP(nn.Module):
    """Exact architecture of `model_ctx_clean_256.pt` and `model_iso_clean_256.pt` distillation checkpoints."""

    def __init__(self, in_dim: int = 1152, hidden_dim: int = 2048, vocab_size: int = 11964):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits_pred": self.net(x)}

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, os.PathLike],
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> "ContextualizedSpatialMLP":
        """Instantiate and load weights from a distillation teacher checkpoint with strict=True."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

        if "net.0.weight" in clean_sd and "net.6.weight" in clean_sd:
            hidden_dim, in_dim = clean_sd["net.0.weight"].shape
            vocab_size = clean_sd["net.6.weight"].shape[0]
            model = cls(in_dim=in_dim, hidden_dim=hidden_dim, vocab_size=vocab_size)
        else:
            model = cls()

        model.load_state_dict(clean_sd, strict=strict)
        if isinstance(ckpt, dict):
            for attr in ("cand_vocab", "stats", "k_cand", "penalty_sm", "penalty_c1", "penalty_c2"):
                if attr in ckpt:
                    setattr(model, attr, ckpt[attr])
        model.to(device)
        model.eval()
        return model


class DistilledSpatialTeacher(ContextualizedSpatialMLP):
    """Canonical Distilled Spatial Teacher (ContextualizedSpatialMLP with from_checkpoint loader)."""
    pass


# copied from arc-agi-3@5396d0d:scripts/dgg2/dgg2_extract_teacher_targets.py
def classify_lexical_token_class(tok_str: str) -> str:
    """Classify a decoded token string into one of the 4 balanced multi-prompt codebook classes.

    Classes:
    - 'emoji_icon': Authentic Unicode emoji symbols.
    - 'unicode_glyph': STRICTLY geometric, box-drawing, block, or arrow symbols.
    - 'cjk_multilingual': Single-token natural language words across all non-ASCII writing systems
                          (Chinese Hanzi, Korean Hangul, Japanese Kana, accented European words).
    - 'english_word': ASCII alphabetic words.
    - 'other': Unclassified punctuation or control artifacts.
    """
    s = str(tok_str).strip()
    if not s:
        return "other"

    # 1. Emoji check
    for ch in s:
        cp = ord(ch)
        if (0x1F000 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF) or cp in (0x2B1B, 0x2B1C, 0x25FE, 0x25FD):
            return "emoji_icon"

    # 2. Strict Unicode Glyph check (ONLY true geometric/box/block/arrow glyphs)
    for ch in s:
        cp = ord(ch)
        if (0x2500 <= cp <= 0x259F) or (0x25A0 <= cp <= 0x25FF) or (0x2190 <= cp <= 0x23FF) or (0x2B00 <= cp <= 0x2BFF):
            return "unicode_glyph"

    # 3. Multilingual Natural Language Word check (Hanzi, Hangul, Kana, accented letters)
    for ch in s:
        cp = ord(ch)
        # Chinese Hanzi
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            return "cjk_multilingual"
        # Korean Hangul (Syllables, Jamo, Compatibility Jamo)
        if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
            return "cjk_multilingual"
        # Japanese Hiragana & Katakana
        if (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            return "cjk_multilingual"
        # Accented Latin, Cyrillic, Greek, Thai, and other Unicode letters
        if unicodedata.category(ch).startswith("L") and not ch.isascii():
            return "cjk_multilingual"

    # 4. English Word check (ASCII alphabetic)
    if s.isascii() and any(c.isalpha() for c in s):
        return "english_word"

    return "other"


# copied from arc-agi-3@5396d0d:scripts/dgg2/colab_runner_phase2_transforms.py
@torch.no_grad()
def extract_teacher_modes_from_hv_batched(
    hv_all: torch.Tensor,
    ctx_ckpt: Union[Dict[str, Any], str, os.PathLike],
    tokenizer: Any,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts 64-mode teacher distributions (`mode_ids` int32 `(N, 256, 64)`,
    `mode_weights` float16 `(N, 256, 64)`) from vision-tower `hv_all` (`(N, 256, 1152)`)
    using `model_ctx_clean_256.pt` with corpus marginal calibration and lexical class stratification.
    """
    if isinstance(ctx_ckpt, (str, os.PathLike)):
        ctx_ckpt = torch.load(ctx_ckpt, map_location="cpu")

    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    cand_vocab = ctx_ckpt["cand_vocab"].to(dev)
    k_cand = int(ctx_ckpt["k_cand"])
    cand_tids = cand_vocab.cpu().tolist()

    m_ctx = DistilledSpatialTeacher(vocab_size=k_cand).to(dev)
    clean_sd = {k.replace("module.", ""): v for k, v in ctx_ckpt["state_dict"].items()}
    m_ctx.load_state_dict(clean_sd, strict=True)
    m_ctx.eval()

    mu_ctx = ctx_ckpt["stats"]["mu"].to(device=dev, dtype=torch.float32)
    std_ctx = ctx_ckpt["stats"]["std"].to(device=dev, dtype=torch.float32)
    penalty_sm = ctx_ckpt["penalty_sm"].to(device=dev, dtype=torch.float32)

    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    classes = [classify_lexical_token_class(decode_fn([t])) for t in cand_tids]
    eng_idx = torch.tensor([i for i, c in enumerate(classes) if c == "english_word"], device=dev)
    cjk_idx = torch.tensor([i for i, c in enumerate(classes) if c == "cjk_multilingual"], device=dev)
    gly_idx = torch.tensor([i for i, c in enumerate(classes) if c == "unicode_glyph"], device=dev)
    emo_idx = torch.tensor([i for i, c in enumerate(classes) if c == "emoji_icon"], device=dev)

    n = int(hv_all.shape[0])
    ids_out_list: List[torch.Tensor] = []
    w_out_list: List[torch.Tensor] = []

    target_mu = 0.18
    beta_step = 0.65

    for s_i in range(0, n, batch_size):
        e_i = min(n, s_i + batch_size)
        b_cur = e_i - s_i
        b_ctx = hv_all[s_i:e_i].to(device=dev, dtype=torch.float32)
        b_ctx_norm = F.normalize((b_ctx.reshape(-1, 1152) - mu_ctx) / std_ctx, dim=-1)
        logits = m_ctx(b_ctx_norm)["logits_pred"]
        p_c1 = F.softmax(logits, dim=-1).reshape(b_cur, 256, k_cand)

        # Corpus calibration + fast vectorized per-image ceiling (6 iterations)
        p_cal = p_c1 / penalty_sm.view(1, 1, k_cand)
        p_cal = p_cal / p_cal.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        for _ in range(6):
            r1 = p_cal.argmax(dim=-1)  # (B, 256)
            one_hot = F.one_hot(r1, num_classes=k_cand).float()  # (B, 256, k_cand)
            freq = one_hot.mean(dim=1)  # (B, k_cand)
            max_freq = freq.max(dim=-1).values
            if float(max_freq.max().item()) <= target_mu:
                break
            over = freq > target_mu
            pen = torch.where(over, (freq / target_mu).pow(beta_step), torch.ones_like(freq))
            p_cal = p_cal / pen.unsqueeze(1)
            p_cal = p_cal / p_cal.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Stratified top-64 extraction (24 eng, 14 cjk, 8 gly, 8 emo, 10 rem)
        p_mat = p_cal.reshape(-1, k_cand)
        m_rows = p_mat.shape[0]

        k_eng = min(24, len(eng_idx))
        k_cjk = min(14, len(cjk_idx))
        k_gly = min(8, len(gly_idx))
        k_emo = min(8, len(emo_idx))

        v_eng, i_eng = torch.topk(p_mat[:, eng_idx], k=k_eng, dim=-1) if k_eng > 0 else (p_mat.new_empty(m_rows, 0), torch.empty(m_rows, 0, dtype=torch.long, device=dev))
        sel_eng = eng_idx[i_eng] if k_eng > 0 else torch.empty(m_rows, 0, dtype=torch.long, device=dev)

        v_cjk, i_cjk = torch.topk(p_mat[:, cjk_idx], k=k_cjk, dim=-1) if k_cjk > 0 else (p_mat.new_empty(m_rows, 0), torch.empty(m_rows, 0, dtype=torch.long, device=dev))
        sel_cjk = cjk_idx[i_cjk] if k_cjk > 0 else torch.empty(m_rows, 0, dtype=torch.long, device=dev)

        v_gly, i_gly = torch.topk(p_mat[:, gly_idx], k=k_gly, dim=-1) if k_gly > 0 else (p_mat.new_empty(m_rows, 0), torch.empty(m_rows, 0, dtype=torch.long, device=dev))
        sel_gly = gly_idx[i_gly] if k_gly > 0 else torch.empty(m_rows, 0, dtype=torch.long, device=dev)

        v_emo, i_emo = torch.topk(p_mat[:, emo_idx], k=k_emo, dim=-1) if k_emo > 0 else (p_mat.new_empty(m_rows, 0), torch.empty(m_rows, 0, dtype=torch.long, device=dev))
        sel_emo = emo_idx[i_emo] if k_emo > 0 else torch.empty(m_rows, 0, dtype=torch.long, device=dev)

        fixed_idx = torch.cat([sel_eng, sel_cjk, sel_gly, sel_emo], dim=-1)
        fixed_w = torch.cat([v_eng, v_cjk, v_gly, v_emo], dim=-1)

        mask = torch.zeros((m_rows, k_cand), dtype=torch.bool, device=dev)
        if fixed_idx.numel() > 0:
            mask.scatter_(1, fixed_idx, True)
        p_rem = p_mat.masked_fill(mask, -1.0)
        k_rem = min(64 - fixed_idx.shape[-1], k_cand - fixed_idx.shape[-1])
        v_rem, i_rem = torch.topk(p_rem, k=max(0, k_rem), dim=-1)

        all_cand_idx = torch.cat([fixed_idx, i_rem], dim=-1)
        all_w = torch.cat([fixed_w, v_rem], dim=-1).clamp_min(1e-8).pow(1.0 / 0.52)
        s_w, s_ord = torch.sort(all_w, dim=-1, descending=True)
        s_cand_idx = torch.gather(all_cand_idx, dim=-1, index=s_ord)

        ids_64 = cand_vocab[s_cand_idx].to(torch.int32).reshape(b_cur, 256, -1).cpu()
        w_64 = (s_w / s_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(torch.float16).reshape(b_cur, 256, -1).cpu()
        ids_out_list.append(ids_64)
        w_out_list.append(w_64)

    del m_ctx
    return torch.cat(ids_out_list, dim=0), torch.cat(w_out_list, dim=0)
