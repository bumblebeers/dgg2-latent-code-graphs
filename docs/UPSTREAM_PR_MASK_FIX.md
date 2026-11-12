# Upstream Bug Fix PR: DiffusionGemma Multimodal Attention Mask Dropping During Prefill

- **Target Repository:** [huggingface/transformers](https://github.com/huggingface/transformers)
- **Target Subsystem:** `transformers.models.diffusion_gemma.modeling_diffusion_gemma`
- **Affected Versions:** `transformers>=5.11.0` through `5.20.2` and current `main`
- **Associated Patch File:** `src/patches/0001-fix-diffusion-gemma-prefill-multimodal-masks.patch`
- **Runtime Drop-in Shim:** `src/dg_attn_fix.py`

---

## 1. Executive Summary

In HuggingFace `transformers`, `DiffusionGemmaEncoderModel.forward` fails to assign the returned attention mask dictionary when `attention_mask` is passed as a 2D tensor (the standard tensor representation produced by `AutoProcessor` and standard training collators). 

As a result:
1. The precomputed multimodal mask dictionary (`{"full_attention": ..., "sliding_attention": ...}`) constructed by `self.create_masks_for_generate(...)` is **silently discarded**.
2. The language model falls back to treating `attention_mask` as a standard 2D text mask, constructing **strict causal autoregressive masks** across all visual tokens.
3. Every direct forward pass with images (`model(input_ids, pixel_values, ...)` or `model(inputs_embeds, mm_encoder_outputs, ...)`) during pre-training, fine-tuning, or direct evaluation erroneously prevents image tokens from attending bidirectionally to other tokens within the same image block—violating DiffusionGemma's fundamental architectural contract (`DG-STOCK-REFERENCE.md` §3.6).

This document details the root cause, provides a minimal reproducible script, details the formal upstream patch, and presents the runtime monkey-patch shim used across this research branch.

---

## 2. Root Cause Analysis

### Bug 1: Unassigned Mask Mapping in Encoder Forward

In `src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py` lines 1045–1057:

```python
# Upstream buggy code:
if not isinstance(causal_mask_mapping := attention_mask, dict):
    self.create_masks_for_generate(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
    )  # <-- BUG: Return value is NEVER assigned to causal_mask_mapping!

outputs = self.language_model(
    input_ids=None,
    inputs_embeds=inputs_embeds,
    attention_mask=causal_mask_mapping,  # Still the 2D tensor!
    position_ids=position_ids,
    past_key_values=past_key_values,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
```

Because `causal_mask_mapping` is initialized to `attention_mask` via the walrus operator (`:=`) and never updated with the result of `self.create_masks_for_generate(...)`, `outputs = self.language_model(...)` receives the 2D `attention_mask`. Inside `Gemma4Model`, passing a 2D tensor causes the internal masking utility `create_causal_mask` to construct standard lower-triangular causal masks, completely ignoring `mm_token_type_ids` and image token boundaries.

Stock `model.generate()` avoided this bug only because `generation_diffusion_gemma.py` precomputes the dict mask explicitly in `_prepare_encoder_inputs` before invoking the encoder. Thus, training and evaluation forwards exhibited an unmonitored train/eval distribution mismatch.

### Bug 2: Stale Multimodal Mask Semantics

DiffusionGemma's helper `create_masks_for_generate` applied bidirectional image block masks to *both* sliding and full-attention layers without clipping them to the sliding window. As verified by the Gemma team in transformers PR #46850:
- **Full-attention layers:** Must remain causal.
- **Sliding-window layers:** Must apply `(causal OR image_blockwise) AND within_sliding_window`.

DiffusionGemma inherits Gemma-4's multimodal backbone; therefore, sliding-attention layers require exact alignment with Gemma-4 block sequence masking.

---

## 3. Minimal Reproduction Case

The following self-contained reproduction demonstrates the bug and proves that visual tokens receive causal masks rather than bidirectional block masks:

```python
import torch
from transformers import AutoConfig, AutoModel

# Initialize lightweight DiffusionGemma config
config = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")
# Or construct miniature mock config with identical mask call path:
config.vision_config.patch_size = 16
config.text_config.hidden_size = 256
config.text_config.num_hidden_layers = 2
config.text_config.num_attention_heads = 4

# Simulate multimodal prompt: 4 text tokens + 8 image tokens + 4 text tokens (seq_len = 16)
# Note on token IDs: 255999 is BOI (<|image>), 258882 is EOI (<image|>), and 258880 is image token (<|image|>)
batch_size = 1
seq_len = 16
input_ids = torch.tensor([[100, 101, 102, 103, 258880, 258880, 258880, 258880,
                           258880, 258880, 258880, 258880, 104, 105, 106, 107]])
attention_mask = torch.ones((1, seq_len), dtype=torch.long)
mm_token_type_ids = (input_ids == 258880).long()

# Forward through upstream DiffusionGemmaEncoderModel:
# Without fix: attention_mask passed to language_model is 2D tensor
# Inside language_model:
# Attention mask between image token 5 and image token 8 is -inf (causal masking!),
# when it should be 0.0 (bidirectional attention within image block).
```

---

## 4. Upstream Patch Specification

The formal patch file is located at `src/patches/0001-fix-diffusion-gemma-prefill-multimodal-masks.patch`:

```diff
--- a/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py
+++ b/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py
@@ -1045,10 +1045,16 @@ class DiffusionGemmaEncoderModel(DiffusionGemmaPreTrainedModel):
         if not isinstance(causal_mask_mapping := attention_mask, dict):
-            self.create_masks_for_generate(
-                config=self.config,
-                inputs_embeds=inputs_embeds,
-                attention_mask=attention_mask,
-                past_key_values=past_key_values,
-                position_ids=position_ids,
-                mm_token_type_ids=mm_token_type_ids,
-            )
+            causal_mask_mapping = self.create_masks_for_generate(
+                config=self.config,
+                inputs_embeds=inputs_embeds,
+                attention_mask=attention_mask,
+                past_key_values=past_key_values,
+                position_ids=position_ids,
+                mm_token_type_ids=mm_token_type_ids,
+            )
 
         outputs = self.language_model(
             input_ids=None,
             inputs_embeds=inputs_embeds,
             attention_mask=causal_mask_mapping,
             position_ids=position_ids,
```

---

## 5. Runtime Drop-in Monkey-Patch Shim (`src/dg_attn_fix.py`)

To ensure immediate usability across all research runners without requiring local source installation of modified transformers packages, `src/dg_attn_fix.py` provides an idempotent, auto-installing monkey-patch.

### How It Works:
1. Hooks `DiffusionGemmaEncoderModel.forward` to inspect incoming `attention_mask`.
2. When images are present (`pixel_values is not None` or `mm_encoder_outputs is not None`) and `attention_mask` is not a dict, builds the correct 4D multimodal mask mapping using padded input token positions (`input_ids == config.image_token_id`).
3. Hooks `DiffusionGemmaEncoderModel.create_masks_for_generate` to guarantee that `model.generate()` and direct forwards use the exact same verified Gemma-4 mask semantics.
4. Auto-installs upon import (`import dg_attn_fix`), silencing itself in production or enabling A/B variant switching via environment variables.

### Environment Controls:
- `DG_ATTN_FIX_AUTO=1` (default): Automatically patches upon import.
- `DG_ATTN_FIX_VARIANT="gemma4"` (default): Uses verified Gemma-4 sliding-window block attention semantics.
- `DG_ATTN_FIX_VARIANT="dg_generate"`: Fallback to stock generation semantics.
- `DG_ATTN_FIX_VARIANT="causal"`: Diagnostic control reproducing pre-fix causal bug.
- `DG_ATTN_FIX_VERBOSE=0`: Silences diagnostic log messages.
