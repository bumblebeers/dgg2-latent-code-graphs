#!/usr/bin/env python3
"""DGG Codebook: Continuous Vocabulary Distribution Derivation.

Constructs multi-modal target probability distributions over DiffusionGemma's
262,144 vocabulary, deriving continuous vocabulary chord embeddings and
salience-weighted targets.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def build_multimodal_target_distribution(
    mode_ids: torch.Tensor,
    salience_weights: torch.Tensor,
    w_thresh: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive normalized multi-modal target distribution with rank-1 crash guard.

    Args:
        mode_ids: (B, L, K) int64 tensor of candidate vocabulary token IDs.
        salience_weights: (B, L, K) raw elicitation salience or probability scores.
        w_thresh: Relative threshold to prune noisy low-probability modes.

    Returns:
        Tuple of (mode_ids, q_tilde) where q_tilde sums to 1.0 along mode dim K.
    """
    mode_ids = mode_ids.long()
    w = salience_weights.float()

    w_max = w.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
    w_norm = w / w_max

    mask = (w_norm >= w_thresh).float()
    mask[..., 0] = 1.0  # Crash guard: Rank-1 mode is ALWAYS preserved

    w_masked = w_norm * mask
    w_sum = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    q_tilde = w_masked / w_sum

    return mode_ids, q_tilde


def compute_expected_vocabulary_embedding(
    q_tilde: torch.Tensor,
    mode_ids: torch.Tensor,
    vocab_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute the continuous expected vocabulary chord e_soft in R^{B x L x D}.

    e_i = sum_{k=1}^K q_tilde_{i, k} * E_lm[mode_ids_{i, k}]

    Args:
        q_tilde: (B, L, K) normalized mode probabilities.
        mode_ids: (B, L, K) integer token IDs.
        vocab_embeddings: (V, D) or nn.Embedding matrix.

    Returns:
        Expected continuous embedding tensor of shape (B, L, D).
    """
    if isinstance(vocab_embeddings, torch.nn.Embedding):
        emb_weights = vocab_embeddings.weight
    else:
        emb_weights = vocab_embeddings

    # Gather embedding vectors for candidate modes: (B, L, K, D)
    flat_ids = mode_ids.reshape(-1)
    gathered = F.embedding(flat_ids, emb_weights)
    b, l, k = mode_ids.shape
    d = emb_weights.shape[-1]
    mode_vectors = gathered.reshape(b, l, k, d)

    # Weighted superposition: (B, L, D)
    q_expanded = q_tilde.unsqueeze(-1)
    e_soft = (q_expanded * mode_vectors.to(q_expanded.dtype)).sum(dim=-2)
    return e_soft.float()


def apply_foreground_salience_boost(
    salience_weights: torch.Tensor,
    is_foreground: torch.Tensor,
    boost_factor: float = 1.5,
) -> torch.Tensor:
    """Boost foreground/edge mode salience when patch contains non-background structure.

    Args:
        salience_weights: (B, L, K) raw salience weights.
        is_foreground: (B, L) boolean mask indicating presence of foreground content.
        boost_factor: Multiplier for non-background modes.

    Returns:
        Adjusted salience weights.
    """
    w = salience_weights.clone()
    fg_mask = is_foreground.unsqueeze(-1).expand_as(w)

    # Modes after rank 0 (photometric background) receive boost if foreground present
    boost = torch.ones_like(w)
    boost[..., 1:] = float(boost_factor)

    w_boosted = torch.where(fg_mask, w * boost, w)
    return w_boosted


def sample_x0_from_distribution(
    mode_ids: torch.Tensor,
    q_tilde: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stochastically sample integer token x0 from continuous mode distribution.

    Used for Sub-Pass A (encoder AR teacher forcing) and Sub-Pass B (D3PM corruption)
    so the model does not latch onto a single deterministic token per patch.

    Args:
        mode_ids: (B, L, K) mode token IDs.
        q_tilde: (B, L, K) normalized probabilities.
        generator: Optional random generator.

    Returns:
        Sampled tokens (B, L) int64.
    """
    b, l, k = q_tilde.shape
    probs_flat = q_tilde.reshape(b * l, k)
    sampled_idx = torch.multinomial(probs_flat, num_samples=1, generator=generator).reshape(b, l, 1)
    x0 = torch.gather(mode_ids, dim=-1, index=sampled_idx).squeeze(-1)
    return x0
