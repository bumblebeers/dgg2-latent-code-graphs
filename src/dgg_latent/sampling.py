#!/usr/bin/env python3
"""DGG Latent: Continuous Latent Diffusion Sampling and Schedules.

Provides sampling algorithms, self-conditioning refinement loops, noise corruption schedules,
and spherical linear interpolation (slerp) for continuous latent diffusion models.
"""
from __future__ import annotations

import copy
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dgg_latent.readout_heads import compute_readout_input


def linear_corruption_schedule(num_steps: int) -> torch.Tensor:
    """Generate linear noise schedule timesteps t in [1.0, 0.0]."""
    return torch.linspace(1.0, 0.0, num_steps)


def cosine_corruption_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Generate cosine noise schedule timesteps."""
    steps = torch.arange(num_steps + 1, dtype=torch.float32)
    f_t = torch.cos(((steps / num_steps + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alpha_bar = f_t / f_t[0]
    timesteps = 1.0 - alpha_bar[:-1]
    return timesteps


def corrupt_discrete_canvas(
    x0: torch.Tensor,
    t: torch.Tensor,
    vocab_size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply D3PM uniform multinomial corruption to discrete token canvas.

    Each token is preserved with probability (1 - t) and replaced with a uniform
    random vocabulary token with probability t:
        x_t = x_0 with prob (1 - t), else unif(0, vocab_size - 1)

    Args:
        x0: Ground truth token tensor of shape (B, L).
        t: Per-example corruption timestep tensor of shape (B,) or (B, 1).
        vocab_size: Size of the discrete vocabulary.
        generator: Optional torch random generator.

    Returns:
        Corrupted discrete canvas of shape (B, L).
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B, 1)

    device = x0.device
    mask = (torch.rand(x0.shape, device=device, generator=generator) < t).long()
    noise = torch.randint(0, vocab_size, x0.shape, device=device, generator=generator)
    return (1 - mask) * x0 + mask * noise


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two continuous latent vectors.

    Args:
        val: Interpolation factor in [0, 1].
        low: Initial latent tensor.
        high: Target latent tensor.

    Returns:
        Interpolated tensor.
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True).clamp(min=1e-8)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True).clamp(min=1e-8)

    omega = torch.acos((low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0))
    so = torch.sin(omega)

    # Fallback to linear interpolation when vectors are collinear
    zero_mask = (so.abs() < 1e-6).float()
    linear_part = (1.0 - val) * low + val * high

    slerp_part = (
        torch.sin((1.0 - val) * omega) / so.clamp(min=1e-8) * low
        + torch.sin(val * omega) / so.clamp(min=1e-8) * high
    )

    return zero_mask * linear_part + (1.0 - zero_mask) * slerp_part


class ContinuousLatentSampler:
    """Sampling manager for continuous latent diffusion with self-conditioning.

    Executes multi-step denoising trajectories with optional self-conditioning
    logit feedback and adaptive stopping based on token entropy and stability.
    """

    def __init__(
        self,
        denoising_steps: int = 48,
        stability_threshold: int = 1,
        confidence_threshold: float = 0.005,
    ):
        self.denoising_steps = denoising_steps
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold

    def is_converged(
        self,
        entropy_history: List[float],
        stability_counter: int,
    ) -> bool:
        """Check if adaptive stopping criteria are satisfied."""
        if not entropy_history:
            return False
        latest_entropy = entropy_history[-1]
        return (
            stability_counter >= self.stability_threshold
            and latest_entropy <= self.confidence_threshold
        )

    def sample_trajectory(
        self,
        step_fn: Callable[[torch.Tensor, Optional[torch.Tensor], float], Tuple[torch.Tensor, torch.Tensor]],
        initial_canvas: torch.Tensor,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run self-conditioned diffusion sampling trajectory.

        Args:
            step_fn: Callable taking (canvas, sc_logits, t) and returning (denoised_canvas, logits).
            initial_canvas: Initial corrupted canvas (B, L).
            verbose: If True, prints step progress.

        Returns:
            Dictionary containing final canvas, final logits, steps used, and stopping reason.
        """
        current_canvas = initial_canvas
        sc_logits: Optional[torch.Tensor] = None
        entropy_history: List[float] = []
        stability_count = 0
        last_argmax = current_canvas

        timesteps = torch.linspace(1.0, 0.0, self.denoising_steps)

        for step_idx, t in enumerate(timesteps):
            denoised_canvas, logits = step_fn(current_canvas, sc_logits, float(t))
            sc_logits = logits.detach()

            # Track stability and entropy
            current_argmax = logits.argmax(dim=-1)
            if torch.equal(current_argmax, last_argmax):
                stability_count += 1
            else:
                stability_count = 0
            last_argmax = current_argmax

            # Mean entropy across tokens
            probs = F.softmax(logits.float(), dim=-1)
            mean_ent = float(-(probs * probs.clamp(min=1e-12).log()).sum(dim=-1).mean().item())
            entropy_history.append(mean_ent)

            current_canvas = denoised_canvas

            if self.is_converged(entropy_history, stability_count) and step_idx > 0:
                return {
                    "canvas": current_canvas,
                    "logits": logits,
                    "steps_used": step_idx + 1,
                    "stopped_by": "stable_and_confident",
                    "entropy_history": entropy_history,
                }

        return {
            "canvas": current_canvas,
            "logits": sc_logits,
            "steps_used": self.denoising_steps,
            "stopped_by": "max_denoising_steps",
            "entropy_history": entropy_history,
        }


# copied from arc-agi-3@5396d0d:scripts/dgg2/dgg2_sample_stock.py

N_TOKENS = 256
STOP_STABLE = "stable_and_confident"
STOP_BUDGET = "max_denoising_steps"


class _GenRecorder:
    """Mutable provenance captured around one stock generate() call."""

    def __init__(
        self,
        milestone_steps: Tuple[int, ...] = (1, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256),
        *,
        record_step_telemetry: bool = True,
    ) -> None:
        self.denoising_steps = 0
        self.denoising_step_calls = 0
        self.per_item_denoising_steps: Optional[torch.Tensor] = None
        self.last_argmax_canvas: Optional[torch.Tensor] = None
        self.last_hidden_state: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.last_processed_logits: Optional[torch.Tensor] = None
        self.criteria_calls = 0
        self.criteria_hits = 0
        self.criteria_stop_ever = False
        self.last_entropy_mean = None
        self.last_stable = None
        self.last_confident = None
        self.instrumented_denoising_step = False
        self.instrumented_criteria = False
        self.instrumented_decoder_hook = False
        self.record_step_telemetry = bool(record_step_telemetry)
        self.milestone_steps = set(int(x) for x in (milestone_steps or ()))
        self.milestone_snapshots: Dict[int, Dict[str, Any]] = {}


class _RecordingCriteria:
    """Duck-typed wrapper around the stock adaptive stopping criteria (records, delegates)."""

    def __init__(self, inner: Any, rec: _GenRecorder):
        self._inner = inner
        self._rec = rec
        self._prev_canvas: Optional[torch.Tensor] = None

    def __call__(self, argmax_canvas: torch.Tensor, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        out = self._inner(argmax_canvas, logits, **kwargs)
        self._rec.criteria_calls += 1
        try:
            self._rec.criteria_hits += int(out.sum().item())
            if bool(out.any()):
                self._rec.criteria_stop_ever = True
        except Exception:
            pass
        return out


@contextmanager
def _instrument_generate(base_model: Any, rec: _GenRecorder):
    """Temporarily wrap per-instance `_denoising_step` / `_prepare_diffusion_stopping_criteria`
    and hook the decoder's final norm to capture `last_hidden_state` (`h_dec`) and `last_logits`
    directly on the final pass of `base_model.generate()`.
    """
    saved: Dict[str, Any] = {}
    missing = object()
    step_cap: Dict[str, torch.Tensor] = {}

    inner = getattr(base_model, "model", base_model)
    if hasattr(inner, "get_base_model"):
        inner = inner.get_base_model()
    inner = getattr(inner, "model", inner)
    dec_mod = getattr(inner, "decoder", None)
    dec_hook_handle = None
    if dec_mod is not None and hasattr(dec_mod, "register_forward_hook"):
        target_mod = getattr(dec_mod, "norm", dec_mod)

        def _dec_hook(_mod: Any, _inp: Any, out: Any) -> None:
            h = getattr(out, "last_hidden_state", None)
            if h is None and isinstance(out, (tuple, list)):
                h = out[0]
            if h is None and isinstance(out, torch.Tensor):
                h = out
            if torch.is_tensor(h):
                step_cap["hidden"] = h[:, -N_TOKENS:, :].detach()

        dec_hook_handle = target_mod.register_forward_hook(_dec_hook)
        rec.instrumented_decoder_hook = True

    dstep = getattr(base_model, "_denoising_step", None)
    if callable(dstep):
        def _wrapped_denoising_step(*args: Any, **kwargs: Any) -> Any:
            step_cap.clear()
            fin_before = kwargs.get("finished_denoising")
            fin_prev = fin_before.detach().clone() if torch.is_tensor(fin_before) else None

            if "decoder_forward" in kwargs and callable(kwargs["decoder_forward"]):
                orig_df = kwargs["decoder_forward"]

                def _wrapped_df(*df_args: Any, **df_kwargs: Any) -> Any:
                    df_out = orig_df(*df_args, **df_kwargs)
                    raw_l = getattr(df_out, "logits", None)
                    if torch.is_tensor(raw_l):
                        step_cap["raw_logits"] = raw_l[:, -N_TOKENS:, :].detach()
                    return df_out

                kwargs = dict(kwargs)
                kwargs["decoder_forward"] = _wrapped_df
            elif len(args) >= 1 and callable(args[0]):
                orig_df = args[0]

                def _wrapped_df(*df_args: Any, **df_kwargs: Any) -> Any:
                    df_out = orig_df(*df_args, **df_kwargs)
                    raw_l = getattr(df_out, "logits", None)
                    if torch.is_tensor(raw_l):
                        step_cap["raw_logits"] = raw_l[:, -N_TOKENS:, :].detach()
                    return df_out

                args = (_wrapped_df,) + args[1:]

            out = dstep(*args, **kwargs)
            rec.denoising_steps += 1
            if fin_prev is not None and fin_prev.ndim == 1:
                active = (~fin_prev).to(torch.long).detach().cpu()
                rec.per_item_denoising_steps = (
                    active if rec.per_item_denoising_steps is None
                    else (rec.per_item_denoising_steps + active)
                )
            try:
                c_cur = out[1].detach()
                if rec.last_argmax_canvas is not None and fin_prev is not None and bool(fin_prev.any()):
                    rec.last_argmax_canvas = torch.where(fin_prev[:, None], rec.last_argmax_canvas, c_cur)
                else:
                    rec.last_argmax_canvas = c_cur
            except Exception:
                pass
            try:
                if len(out) > 2 and torch.is_tensor(out[2]):
                    pl_cur = out[2].detach()
                    if rec.last_processed_logits is not None and fin_prev is not None and bool(fin_prev.any()):
                        rec.last_processed_logits = torch.where(fin_prev[:, None, None], rec.last_processed_logits, pl_cur)
                    else:
                        rec.last_processed_logits = pl_cur
            except Exception:
                pass
            if "hidden" in step_cap:
                h_cur = step_cap["hidden"]
                if rec.last_hidden_state is not None and fin_prev is not None and bool(fin_prev.any()):
                    rec.last_hidden_state = torch.where(fin_prev[:, None, None], rec.last_hidden_state, h_cur)
                else:
                    rec.last_hidden_state = h_cur
            if "raw_logits" in step_cap:
                l_cur = step_cap["raw_logits"]
                if rec.last_logits is not None and fin_prev is not None and bool(fin_prev.any()):
                    rec.last_logits = torch.where(fin_prev[:, None, None], rec.last_logits, l_cur)
                else:
                    rec.last_logits = l_cur
            return out

        saved["_denoising_step"] = base_model.__dict__.get("_denoising_step", missing)
        base_model._denoising_step = _wrapped_denoising_step
        rec.instrumented_denoising_step = True

    prep = getattr(base_model, "_prepare_diffusion_stopping_criteria", None)
    if callable(prep):
        def _wrapped_prepare(generation_config: Any) -> Any:
            criteria = prep(generation_config)
            if criteria is None:
                return None
            return _RecordingCriteria(criteria, rec)
        saved["_prepare_diffusion_stopping_criteria"] = base_model.__dict__.get(
            "_prepare_diffusion_stopping_criteria", missing)
        base_model._prepare_diffusion_stopping_criteria = _wrapped_prepare
        rec.instrumented_criteria = True

    try:
        yield rec
    finally:
        if dec_hook_handle is not None:
            dec_hook_handle.remove()
        for name, old in saved.items():
            if old is missing:
                base_model.__dict__.pop(name, None)
            else:
                setattr(base_model, name, old)


def load_stock_generation_config(model_path: Union[str, Path, os.PathLike]) -> Tuple[object, dict]:
    """Load the checkpoint's `generation_config.json` as a `DiffusionGemmaGenerationConfig`."""
    path = Path(str(model_path)).expanduser()
    cfg_file = path / "generation_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"[stock] {cfg_file} not found — stock sampling needs generation_config.json")

    try:
        from transformers.models.diffusion_gemma import DiffusionGemmaGenerationConfig
        gc = DiffusionGemmaGenerationConfig.from_pretrained(str(path))
    except Exception:
        import json
        with open(cfg_file) as f:
            data = json.load(f)
        gc = type("DiffusionGemmaGenerationConfig", (), data)()

    sampler = getattr(gc, "sampler_config", None)
    entropy_bound = getattr(sampler, "entropy_bound", None) if sampler else None
    if entropy_bound is None and hasattr(gc, "entropy_bound"):
        entropy_bound = gc.entropy_bound

    prov = {
        "source": str(cfg_file),
        "sampler": "EntropyBoundSampler",
        "entropy_bound": float(entropy_bound) if entropy_bound is not None else 0.1,
        "max_denoising_steps": int(getattr(gc, "max_denoising_steps", 48)),
        "t_max": float(getattr(gc, "t_max", 0.8)),
        "t_min": float(getattr(gc, "t_min", 0.4)),
        "stopping": "StableAndConfidentStoppingCriteria",
        "stability_threshold": int(getattr(gc, "stability_threshold", 1)),
        "confidence_threshold": float(getattr(gc, "confidence_threshold", 0.005)),
        "eos_token_id": (list(gc.eos_token_id) if isinstance(getattr(gc, "eos_token_id", None), (list, tuple))
                         else getattr(gc, "eos_token_id", None)),
        "pad_token_id": getattr(gc, "pad_token_id", None),
        "checkpoint_max_new_tokens": getattr(gc, "max_new_tokens", 256),
    }
    return gc, prov


def _chunk_dist_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(n, V) float32 logits -> ((n,) max prob, (n,) entropy nats) of softmax(x)."""
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    maxp = (e / s).max(dim=-1).values
    log_z = m.squeeze(-1) + torch.log(s.squeeze(-1))
    mean_x = (e * x).sum(dim=-1) / s.squeeze(-1)
    return maxp, log_z - mean_x


def stock_generate(
    base_model: Any,
    inputs: dict,
    *,
    max_new_tokens: int = N_TOKENS,
    gen_config: Any = None,
    recorder: Optional[_GenRecorder] = None,
) -> dict:
    """Run DG's stock generate(**inputs, max_new_tokens=256) and capture provenance."""
    if int(max_new_tokens) <= 0 or int(max_new_tokens) % N_TOKENS:
        raise ValueError(f"max_new_tokens must be a positive multiple of {N_TOKENS}, got {max_new_tokens}")
    rec = recorder if recorder is not None else _GenRecorder()
    model_cfg = gen_config if gen_config is not None else getattr(base_model, "generation_config", None)
    max_steps = int(getattr(model_cfg, "max_denoising_steps", 0) or 0)
    pad_id = getattr(model_cfg, "pad_token_id", None)

    with _instrument_generate(base_model, rec):
        out = base_model.generate(**dict(inputs), max_new_tokens=int(max_new_tokens))

    if hasattr(out, "sequences"):
        sequences = out.sequences
        tpf_t = getattr(out, "tokens_per_forward", None)
    else:
        sequences = out
        tpf_t = None
    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = sequences[:, prompt_len:]
    n_canvases = int(math.ceil(int(max_new_tokens) / N_TOKENS))
    if new_tokens.shape[1] < N_TOKENS:
        raise RuntimeError(f"[stock] generate returned {new_tokens.shape[1]} new tokens (expected >= {N_TOKENS})")
    seq_canvas = new_tokens[:, -N_TOKENS:].detach()
    canvas = seq_canvas
    canvas_source = "sequences_last_canvas"
    captured = rec.last_argmax_canvas
    bsz = int(sequences.shape[0])
    if torch.is_tensor(captured) and tuple(captured.shape) == (bsz, N_TOKENS):
        canvas = captured.to(seq_canvas.device)
        canvas_source = "denoising_step_argmax"
    valid_new = (int((new_tokens != pad_id).sum()) if pad_id is not None
                 else int(new_tokens.shape[1]))
    tpf = float(tpf_t.reshape(-1)[0]) if torch.is_tensor(tpf_t) else None
    denoising_steps = int(rec.denoising_steps) if rec.instrumented_denoising_step else None
    steps_source = "instrumented_denoising_step" if denoising_steps is not None else None
    if denoising_steps is None and tpf:
        denoising_steps = int(round(valid_new / tpf))
        steps_source = "derived_tokens_per_forward"
    if denoising_steps is None and max_steps:
        denoising_steps = max_steps
        steps_source = "max_denoising_steps_fallback"
    eos_ids = getattr(model_cfg, "eos_token_id", None)
    eos_found = False
    if eos_ids is not None:
        eos = torch.tensor(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids],
                           device=new_tokens.device)
        eos_found = bool(torch.isin(new_tokens, eos).any())
    if denoising_steps is not None and max_steps:
        stopped_by = STOP_BUDGET if int(denoising_steps) >= max_steps else STOP_STABLE
    else:
        stopped_by = "unknown"
    return {
        "sequences": sequences.detach(),
        "prompt_len": prompt_len,
        "canvas": canvas,
        "hidden": rec.last_hidden_state,
        "logits": rec.last_logits if rec.last_logits is not None else rec.last_processed_logits,
        "processed_logits": rec.last_processed_logits,
        "canvas_source": canvas_source,
        "canvas_matches_sequences": bool(torch.equal(canvas, seq_canvas)),
        "n_canvases": n_canvases,
        "denoising_steps": denoising_steps,
        "denoising_steps_source": steps_source,
        "stopped_by": stopped_by,
        "tokens_per_forward": tpf,
        "valid_new_tokens": valid_new,
        "eos_found": eos_found,
        "generation_config": gen_config,
        "recorder": rec,
    }


def stock_generate_with_readout(
    base_model: Any,
    inputs: dict,
    *,
    heads: Optional[Union[torch.nn.Module, Dict[str, torch.nn.Module]]] = None,
    readout_mode: str = "dual_gated",
    lm_weight: Optional[torch.Tensor] = None,
    readout_tau: float = 1.0,
    hv_mean: Optional[torch.Tensor] = None,
    max_new_tokens: int = N_TOKENS,
    gen_config: Any = None,
    recorder: Optional[_GenRecorder] = None,
    cfg_scale: float = 1.5,
    eval_seeds: Optional[Union[List[int], Tuple[int, ...]]] = None,
    include_dist_stats: bool = False,
) -> dict:
    """Canonical generation readout imported by every runner.

    Built directly on `_instrument_generate` via `stock_generate`:
    `h_dec` and `logits` come from the final active self-conditioned denoising step
    of `base_model.generate()`, respecting per-example `finished_denoising`.
    """
    with torch.no_grad():
        res = stock_generate(
            base_model,
            inputs,
            max_new_tokens=max_new_tokens,
            gen_config=gen_config,
            recorder=recorder,
        )
        h_dec = res.get("hidden")
        logits = res.get("logits")
        if h_dec is None:
            raise RuntimeError(
                "[stock_generate_with_readout] _instrument_generate failed to capture last_hidden_state (h_dec)"
            )
        if readout_mode in ("logits", "dual_gated") and logits is None:
            raise RuntimeError(
                f"[stock_generate_with_readout] readout_mode={readout_mode!r} requires final SC logits, got None"
            )

        readout_in = compute_readout_input(
            h_dec=h_dec,
            logits=logits,
            readout_mode=readout_mode,
            lm_weight=lm_weight,
            tau=float(readout_tau),
        )
        res["readout_input"] = readout_in
        res["readout_source"] = "final_sc_denoising_step"

        def _apply_hv_mean(hv_t: torch.Tensor) -> torch.Tensor:
            if hv_mean is None:
                return hv_t
            hm = hv_mean.to(device=hv_t.device, dtype=torch.float32)
            return hv_t + hm

        def _eval_one_head(h_mod: torch.nn.Module):
            h_mod.eval()
            if hasattr(h_mod, "sample"):
                seeds = tuple(eval_seeds) if eval_seeds else (42,)
                draws_res = [
                    h_mod.sample(readout_in, cfg_scale=float(cfg_scale), seed=int(s)).float()
                    for s in seeds
                ]
                draws_full = [_apply_hv_mean(d) for d in draws_res]
                return draws_res[0], draws_full[0], draws_full, 0.0
            hv_res_t = h_mod(readout_in).float()
            hv_full_t = _apply_hv_mean(hv_res_t)
            g_val = float(getattr(h_mod, "last_gate_mean", 0.0))
            return hv_res_t, hv_full_t, None, g_val

        if isinstance(heads, dict):
            preds_by_head: Dict[str, torch.Tensor] = {}
            preds_res_by_head: Dict[str, torch.Tensor] = {}
            multiseed_by_head: Dict[str, Optional[List[torch.Tensor]]] = {}
            gate_by_head: Dict[str, float] = {}
            for hk, h_mod in heads.items():
                hv_r, hv_f, ms_f, g_v = _eval_one_head(h_mod)
                preds_res_by_head[hk] = hv_r
                preds_by_head[hk] = hv_f
                multiseed_by_head[hk] = ms_f
                gate_by_head[hk] = g_v
            res["preds_by_head"] = preds_by_head
            res["preds_res_by_head"] = preds_res_by_head
            res["multiseed_by_head"] = multiseed_by_head
            res["gate_by_head"] = gate_by_head
        elif heads is not None:
            hv_r, hv_f, ms_f, g_v = _eval_one_head(heads)
            res["hv_res"] = hv_r
            res["hv_full"] = hv_f
            res["multiseed_hv"] = ms_f
            res["gate_mean"] = g_v

        if include_dist_stats and logits is not None:
            x = logits[0].float() if logits.dim() == 3 else logits.float()
            maxp, ent = [], []
            for start in range(0, x.shape[0], 64):
                mp, en = _chunk_dist_stats(x[start:start + 64])
                maxp.append(mp)
                ent.append(en)
            res["readout_maxprob"] = float(torch.cat(maxp).mean())
            res["readout_entropy"] = float(torch.cat(ent).mean())
        elif include_dist_stats:
            res["readout_maxprob"] = 0.0
            res["readout_entropy"] = 0.0

    return res
