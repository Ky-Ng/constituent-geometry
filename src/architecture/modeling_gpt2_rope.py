"""GPT2-style decoder-only Transformer with RoPE on Q/K (experiment 08).

Block design choices (faithful to GPT2 except where noted):
* **Pre-LayerNorm**: ``x + dropout(sublayer(LN(x)))`` per sublayer. Differs from
  the post-LN Vaswani decoder in ``modeling_vaswani_rope.py``.
* **GELU** activation in the FFN. (GPT2's exact "approximate" GELU vs. the
  exact one is a non-issue here; we let ``ACT2FN[config.activation]`` pick.)
* **Biases on linear layers**. GPT2 fuses ``Conv1D(in, 3*in)`` for QKV; we keep
  q/k/v split for clarity and parity with ``modeling_vaswani_rope.py``. The math
  is identical -- Conv1D is just ``Linear`` with the weight transposed.
* **Final LayerNorm** before the LM head, as in GPT2.
* **Tied** input embeddings and LM head (HF default; GPT2 does this).
* **No learned/sinusoidal positional embedding** -- the only positional signal
  is RoPE applied to Q/K inside every self-attention layer. This is the
  intended difference from a vanilla GPT2.
* **No KV cache** -- same deliberate simplification as the Vaswani code: the
  toy sequences are short, and generation recomputes the full prefix each
  step. Keeps the model file readable.

What this is NOT:
* Not a Llama variant: no RMSNorm, no SwiGLU, biases ARE present.
* Not the original Vaswani decoder: pre-LN (not post-LN), GELU (not ReLU),
  causal-only stack (no cross-attention).
* Not HuggingFace's ``GPT2LMHeadModel`` with a monkey-patched RoPE -- we
  hand-write the model so the positional-encoding swap is obvious and so it
  follows the project pattern of self-contained custom models.

To copy: rename this file to ``modeling_gpt2_rope.py``.

Smoke test (see ``if __name__ == "__main__"`` at the bottom):
    uv run python src/architecture/modeling_gpt2_rope.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from architecture.configuration_gpt2_rope import GPT2RoPEConfig


# =============================================================================
# Mask helpers (identical to modeling_vaswani.py). Attention masks are *additive*.
# =============================================================================
def _make_causal_mask(tgt_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    return torch.triu(mask, diagonal=1)


def _expand_padding_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    bsz, src_len = mask.shape
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


# =============================================================================
# RoPE. Half-rotation convention -- identical math to
# modeling_vaswani_rope.RotaryEmbedding. Pulled in as a fresh class rather than
# imported so this file stands alone if the Vaswani code is later refactored.
# =============================================================================
class RotaryEmbedding(nn.Module):
    """Half-rotated RoPE: returns ``(cos, sin)`` tables for a given prefix length.

    See ``modeling_vaswani_rope.RotaryEmbedding`` for the derivation; the
    implementation here is byte-identical.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)        # [max_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)         # [max_len, head_dim]
        # persistent=True for the same reason as the sinusoidal PE buffer in
        # modeling_vaswani.py: meta-device loading via from_pretrained leaves
        # non-persistent buffers uninitialized, which produces silent NaNs.
        self.register_buffer("cos", emb.cos(), persistent=True)
        self.register_buffer("sin", emb.sin(), persistent=True)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]   # each [seq_len, head_dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# =============================================================================
# Causal multi-head self-attention with RoPE on Q/K.
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2RoPEConfig, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scaling = self.head_dim**-0.5
        # GPT2 fuses Q/K/V into one Conv1D. We keep them split to mirror
        # modeling_vaswani_rope.py and make the RoPE application sites obvious.
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Plain Python attribute, NOT a submodule. nn.Module.__setattr__ would
        # auto-register an nn.Module value as a child and add duplicate
        # rotary_emb.cos / rotary_emb.sin entries to this layer's state_dict --
        # safetensors then refuses to save. The buffer is owned canonically by
        # GPT2RoPEModel.rotary_emb; the MHA just needs a reference.
        object.__setattr__(self, "rotary_emb", rotary_emb)

    def _split_heads(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        return x.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        bsz, tgt_len, _ = hidden_states.size()
        q = self._split_heads(self.q_proj(hidden_states) * self.scaling, bsz)
        k = self._split_heads(self.k_proj(hidden_states), bsz)
        v = self._split_heads(self.v_proj(hidden_states), bsz)

        cos, sin = self.rotary_emb(tgt_len)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-1, -2))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(self.dropout(weights), v)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, -1)
        attn = self.out_proj(attn)
        return attn, (weights if output_attentions else None)


# =============================================================================
# FFN: Linear -> GELU -> Linear.
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, config: GPT2RoPEConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# =============================================================================
# GPT2-style pre-LN block: x + drop(attn(LN(x))); x + drop(ffn(LN(x))).
# =============================================================================
class GPT2Block(nn.Module):
    def __init__(self, config: GPT2RoPEConfig, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config, rotary_emb=rotary_emb)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_out, attn = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.dropout(self.mlp(self.ln_2(hidden_states)))
        return hidden_states, attn


# =============================================================================
# HF-facing models.
# =============================================================================
class GPT2RoPEPreTrainedModel(PreTrainedModel):
    """Shared base. We deliberately do NOT override ``_init_weights`` -- see the
    docstring of ``VaswaniPreTrainedModel`` for the gotcha (.data bypasses the
    HF guard and re-initializes loaded weights)."""

    config_class = GPT2RoPEConfig
    base_model_prefix = "transformer"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False


class GPT2RoPEModel(GPT2RoPEPreTrainedModel):
    """Embedding + N transformer blocks + final LayerNorm. No LM head."""

    def __init__(self, config: GPT2RoPEConfig) -> None:
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        head_dim = config.d_model // config.n_heads
        # One RoPE table shared across every layer.
        self.rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            GPT2Block(config, self.rotary_emb) for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.wte = value

    def forward(self, input_ids=None, attention_mask=None, output_attentions=None,
                return_dict=None, **kwargs) -> BaseModelOutputWithPast:
        output_attentions = bool(output_attentions)
        bsz, tgt_len = input_ids.shape
        hidden = self.drop(self.wte(input_ids))

        causal = _make_causal_mask(tgt_len, hidden.dtype, hidden.device)[None, None, :, :]
        if attention_mask is not None:
            causal = causal + _expand_padding_mask(attention_mask, hidden.dtype, tgt_len)

        attns = () if output_attentions else None
        for block in self.h:
            hidden, attn = block(hidden, attention_mask=causal, output_attentions=output_attentions)
            if output_attentions:
                attns += (attn,)
        hidden = self.ln_f(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden, attentions=attns)


class GPT2RoPEForCausalLM(GPT2RoPEPreTrainedModel, GenerationMixin):
    """Decoder-only causal LM with a tied LM head and ``.generate()`` support."""

    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: GPT2RoPEConfig) -> None:
        super().__init__(config)
        self.transformer = GPT2RoPEModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_attentions=None, return_dict=None, **kwargs) -> CausalLMOutputWithPast:
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # Causal-LM shift: position t predicts labels[t+1]. We do the shift
            # here so the caller can pass ``labels = input_ids`` (with positions
            # to ignore set to -100). For experiment 08, the run script masks
            # everything up to and including <sot> with -100, so the first
            # un-masked prediction uses input <sot> to predict hf_1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache: feed the full prefix each step. Toy sequences are short
        # enough that the O(T^2) recompute is fine and the code stays readable.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}


if __name__ == "__main__":
    # Smoke test: build, label-mask, forward, generate, round-trip a checkpoint.
    cfg = GPT2RoPEConfig(
        vocab_size=20, d_model=32, n_heads=4, n_layers=2, d_ff=64,
        max_position_embeddings=32, sot_token_id=4,
    )
    m = GPT2RoPEForCausalLM(cfg).eval()

    # Fake sequence: <bos> a b <sot> c d <eos> (ids inside vocab_size=20).
    ids = torch.tensor([[1, 5, 6, 4, 7, 8, 2]])
    labels = ids.clone()
    labels[:, :4] = -100  # ignore <bos>, a, b, <sot> -- first un-masked is c.
    out = m(input_ids=ids, labels=labels)
    print(f"loss={out.loss.item():.3f}  logits={tuple(out.logits.shape)}")

    # Generate from the prompt `<bos> a b <sot>`.
    gen = m.generate(input_ids=ids[:, :4], max_new_tokens=5,
                     do_sample=False, pad_token_id=0, eos_token_id=2)
    print(f"generated ids: {gen.tolist()}")

    m.save_pretrained("data/ckpt_gpt2_rope_smoke")
    m2 = GPT2RoPEForCausalLM.from_pretrained("data/ckpt_gpt2_rope_smoke")
    print(
        f"reloaded OK, tied lm_head==wte: "
        f"{m2.lm_head.weight.data_ptr() == m2.get_input_embeddings().weight.data_ptr()}"
    )
