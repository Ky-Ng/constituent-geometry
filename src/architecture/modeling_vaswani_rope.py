"""Vaswani encoder-decoder with RoPE instead of fixed sinusoidal positional encoding.

What changes vs. ``modeling_vaswani.py``:
* No sinusoidal positional table added at the input. Token embeddings flow into
  the encoder/decoder unmodified (other than the sqrt(d_model) scale we keep
  for parity with the rest of the Vaswani recipe).
* RoPE rotates Q and K *inside self-attention*, once per layer, by an angle
  proportional to the token's position. Cross-attention is unchanged — there is
  no shared position frame between the encoder and decoder sequences, so RoPE
  does not apply there.
* Everything else (post-LN sublayers, weight tying, no KV cache, paper-faithful
  ReLU FFN) is inherited verbatim from the original file by importing the
  helpers we don't need to touch.

Implementation notes:
* **Half-rotation convention** (Llama / GPT-NeoX): for a head dim of size d,
  pair coordinate i with i + d/2 rather than the original RoFormer's adjacent
  (0,1),(2,3),... interleaving. The two are equivalent up to a fixed
  permutation of the head_dim axis, and the half form composes with the
  standard Q/K projections without any gather op.
* ``head_dim`` must be even (RoPE needs to split the dim in half).
* **Shared rotary table: bypass submodule auto-registration.**
  ``VaswaniRoPEModel`` owns a single ``RotaryEmbedding`` and hands the same
  Python object to every ``MultiHeadAttention.self_attn``. A naive
  ``self.rotary_emb = rotary_emb`` triggers ``nn.Module.__setattr__``, which
  auto-registers ``rotary_emb`` as a child submodule of every attention
  layer. The ``cos`` / ``sin`` buffers then appear in ``state_dict`` at both
  the canonical ``model.rotary_emb.cos`` path AND under every layer
  (``encoder.layers.0.self_attn.rotary_emb.cos`` etc.). Safetensors refuses
  to save shared tensors that aren't declared in ``_tied_weights_keys``, so
  the first checkpoint save crashes the run. ``_tied_weights_keys`` is also
  not a usable escape hatch here: HF's ``tie_weights()`` resolves each alias
  via ``get_parameter()``, which raises ``AttributeError`` on buffers.
  We sidestep both by storing the reference via ``object.__setattr__`` so
  the attribute lives only in ``__dict__`` — the canonical owner remains
  ``VaswaniRoPEModel.rotary_emb`` and the layers hold a plain Python
  reference for ``forward()``. ``layer.to(device)`` still moves the buffers
  correctly: the model-level ``rotary_emb`` is registered on
  ``VaswaniRoPEModel`` and migrated by ``model.to(...)``; the layer-level
  reference is the same Python object so it sees the post-move tensors
  automatically.

To copy: rename this file to ``modeling_vaswani_rope.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from architecture.configuration_vaswani import VaswaniConfig
from architecture.modeling_vaswani import (
    FeedForward,
    _expand_padding_mask,
    _make_causal_mask,
    shift_tokens_right,
)


# =============================================================================
# RoPE: precomputed cos/sin tables; rotate Q/K by position inside attention.
# =============================================================================
class RotaryEmbedding(nn.Module):
    """Half-rotated RoPE: returns ``(cos, sin)`` tables for a given prefix length.

    Frequency schedule matches sinusoidal PE: ``theta_i = base^(-2i/d)`` for
    ``i = 0..d/2-1``. We duplicate the per-pair frequencies along the head_dim
    axis so the rotation can be written without any interleaving:

        x_rot = x * cos + rotate_half(x) * sin
        rotate_half([x1, x2]) = [-x2, x1]    # x1, x2 are first/second halves
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        # inv_freq[i] = 1 / base^(2i/d), i = 0..d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)        # [max_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)         # [max_len, head_dim] (half-rotation duplication)
        # persistent=True for the same reason as the sinusoidal PE buffer in
        # modeling_vaswani.py: meta-device loading via from_pretrained leaves
        # non-persistent buffers uninitialized, which produces silent NaNs.
        self.register_buffer("cos", emb.cos(), persistent=True)
        self.register_buffer("sin", emb.sin(), persistent=True)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]   # each [seq_len, head_dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[..., d]`` -> ``[..., d]`` with halves swapped and the first half negated."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, head_dim], cos/sin: [T, head_dim] -> broadcasts over [B, H].
    return x * cos + _rotate_half(x) * sin


# =============================================================================
# Multi-head attention. Identical to the original, except Q and K are rotated
# by RoPE when ``rotary_emb`` is provided. Cross-attention modules pass None.
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config: VaswaniConfig, rotary_emb: RotaryEmbedding | None) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # Plain Python attribute, NOT a submodule. ``nn.Module.__setattr__``
        # would auto-register an nn.Module value as a child and add a duplicate
        # ``rotary_emb.cos`` / ``rotary_emb.sin`` path to this layer's
        # ``state_dict``. The buffer is already owned canonically by
        # ``VaswaniRoPEModel.rotary_emb`` -- see module docstring.
        object.__setattr__(self, "rotary_emb", rotary_emb)

    def _split_heads(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        return x.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, tgt_len, _ = hidden_states.size()
        kv = hidden_states if key_value_states is None else key_value_states

        q = self._split_heads(self.q_proj(hidden_states) * self.scaling, bsz)
        k = self._split_heads(self.k_proj(kv), bsz)
        v = self._split_heads(self.v_proj(kv), bsz)

        if self.rotary_emb is not None:
            # Self-attention only: kv is hidden_states, so q and k share length.
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
# Layers: encoder = self-attn (RoPE) + FFN
#         decoder = self-attn (RoPE) + cross-attn (no RoPE) + FFN
# =============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, config: VaswaniConfig, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config, rotary_emb=rotary_emb)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        residual = hidden_states
        hidden_states, attn = self.self_attn(hidden_states, attention_mask=attention_mask,
                                             output_attentions=output_attentions)
        hidden_states = self.self_attn_layer_norm(residual + self.dropout(hidden_states))

        residual = hidden_states
        hidden_states = self.final_layer_norm(residual + self.dropout(self.ffn(hidden_states)))
        return hidden_states, attn


class DecoderLayer(nn.Module):
    def __init__(self, config: VaswaniConfig, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config, rotary_emb=rotary_emb)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # rotary_emb=None disables RoPE on cross-attention.
        self.cross_attn = MultiHeadAttention(config, rotary_emb=None)
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states,
                encoder_attention_mask, output_attentions=False):
        residual = hidden_states
        hidden_states, self_attn = self.self_attn(hidden_states, attention_mask=attention_mask,
                                                  output_attentions=output_attentions)
        hidden_states = self.self_attn_layer_norm(residual + self.dropout(hidden_states))

        residual = hidden_states
        hidden_states, cross_attn = self.cross_attn(
            hidden_states, key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, output_attentions=output_attentions,
        )
        hidden_states = self.cross_attn_layer_norm(residual + self.dropout(hidden_states))

        residual = hidden_states
        hidden_states = self.final_layer_norm(residual + self.dropout(self.ffn(hidden_states)))
        return hidden_states, self_attn, cross_attn


# =============================================================================
# Stacks. Input pipeline is just embed -> scale -> dropout; no PE added.
# =============================================================================
class VaswaniRoPEEncoder(nn.Module):
    def __init__(self, config: VaswaniConfig, embed_tokens: nn.Embedding,
                 rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderLayer(config, rotary_emb) for _ in range(config.n_layers))

    def forward(self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs):
        hidden = self.dropout(self.embed_tokens(input_ids) * self.embed_scale)
        mask = None
        if attention_mask is not None:
            mask = _expand_padding_mask(attention_mask, hidden.dtype, input_ids.size(1))

        attentions = () if output_attentions else None
        for layer in self.layers:
            hidden, attn = layer(hidden, mask, output_attentions=output_attentions)
            if output_attentions:
                attentions += (attn,)
        return BaseModelOutput(last_hidden_state=hidden, attentions=attentions)


class VaswaniRoPEDecoder(nn.Module):
    def __init__(self, config: VaswaniConfig, embed_tokens: nn.Embedding,
                 rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderLayer(config, rotary_emb) for _ in range(config.n_layers))

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, output_attentions=False):
        tgt_len = input_ids.size(1)
        hidden = self.dropout(self.embed_tokens(input_ids) * self.embed_scale)

        self_mask = _make_causal_mask(tgt_len, hidden.dtype, hidden.device)[None, None, :, :]
        if attention_mask is not None:
            self_mask = self_mask + _expand_padding_mask(attention_mask, hidden.dtype, tgt_len)
        cross_mask = (
            _expand_padding_mask(encoder_attention_mask, hidden.dtype, tgt_len)
            if encoder_attention_mask is not None else None
        )

        self_attns = () if output_attentions else None
        cross_attns = () if output_attentions else None
        for layer in self.layers:
            hidden, s_attn, c_attn = layer(hidden, self_mask, encoder_hidden_states,
                                           cross_mask, output_attentions=output_attentions)
            if output_attentions:
                self_attns += (s_attn,)
                cross_attns += (c_attn,)
        return hidden, self_attns, cross_attns


# =============================================================================
# HF-facing models. Identical scaffolding to modeling_vaswani.py; only the
# encoder/decoder modules and the new RoPE table differ.
# =============================================================================
class VaswaniRoPEPreTrainedModel(PreTrainedModel):
    config_class = VaswaniConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False


class VaswaniRoPEModel(VaswaniRoPEPreTrainedModel):
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        # One RoPE table shared across every self-attention layer (encoder + decoder).
        # head_dim and max_len are identical on both sides, so a single buffer pair
        # saves storage in the checkpoint and avoids any drift between the two
        # stacks if max_position_embeddings is later changed.
        head_dim = config.d_model // config.n_heads
        self.rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings)
        self.encoder = VaswaniRoPEEncoder(config, self.shared, self.rotary_emb)
        self.decoder = VaswaniRoPEDecoder(config, self.shared, self.rotary_emb)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.shared = value
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value

    def get_encoder(self) -> VaswaniRoPEEncoder:
        return self.encoder

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, encoder_outputs=None, output_attentions=None,
                return_dict=None, **kwargs) -> Seq2SeqModelOutput:
        output_attentions = bool(output_attentions)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                           output_attentions=output_attentions)
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        dec_hidden, dec_self, dec_cross = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask, output_attentions=output_attentions,
        )
        return Seq2SeqModelOutput(
            last_hidden_state=dec_hidden,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            decoder_attentions=dec_self,
            cross_attentions=dec_cross,
            encoder_attentions=encoder_outputs.attentions,
        )


class VaswaniRoPEForConditionalGeneration(VaswaniRoPEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.shared.weight"}

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__(config)
        self.model = VaswaniRoPEModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, encoder_outputs=None, labels=None,
                output_attentions=None, return_dict=None, **kwargs) -> Seq2SeqLMOutput:
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, output_attentions=output_attentions,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_outputs=None,
                                      attention_mask=None, **kwargs):
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }


if __name__ == "__main__":
    cfg = VaswaniConfig(vocab_size=15, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = VaswaniRoPEForConditionalGeneration(cfg).eval()

    src = torch.tensor([[1, 5, 6, 7, 2]])
    labels = torch.tensor([[8, 9, 2]])
    out = model(input_ids=src, labels=labels)
    print(f"loss={out.loss.item():.3f}  logits={tuple(out.logits.shape)}")

    gen = model.generate(input_ids=src, max_new_tokens=5)
    print(f"generated ids: {gen.tolist()}")

    model.save_pretrained("data/ckpt_smoketest_rope")
    reloaded = VaswaniRoPEForConditionalGeneration.from_pretrained("data/ckpt_smoketest_rope")
    print(f"reloaded OK, tied lm_head==shared: "
          f"{reloaded.lm_head.weight.data_ptr() == reloaded.get_input_embeddings().weight.data_ptr()}")
