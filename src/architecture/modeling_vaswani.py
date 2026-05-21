"""A from-scratch "Attention is All You Need" encoder-decoder, HF-compatible.

We hand-write every piece of the original Transformer — sinusoidal positional
encoding, multi-head attention, the post-LayerNorm sublayers, the encoder and
decoder stacks — but we mount them on HuggingFace base classes so we inherit, for
free, exactly the behaviour you asked for:

* ``from_pretrained(dir)`` / ``save_pretrained(dir)`` — checkpoint loading/saving
  into ``model.safetensors`` + ``config.json`` + ``generation_config.json``, the
  same artifact set as the gemma repo you looked at.
* ``.generate(...)`` — via ``GenerationMixin``, for HI -> HF decoding.

Faithfulness notes (vs. the 2017 paper):
* fixed **sinusoidal** positions, no learned position params;
* **post-LayerNorm** residual blocks: ``LayerNorm(x + Sublayer(x))``;
* embeddings scaled by ``sqrt(d_model)`` and **tied** to the output projection;
* ReLU feed-forward (configurable via ``config.activation``).

Deliberate simplification: **no KV cache.** Generation recomputes the decoder
over the full prefix at each step. For the toy grammar (short sentences) this is
fine and keeps the code readable; a Cache implementation can be added later.

To copy: rename this file to ``modeling_vaswani.py``.

Quick smoke test once copied (see bottom of file for the runnable version):
    from architecture.modeling_vaswani import VaswaniForConditionalGeneration
    from architecture.configuration_vaswani import VaswaniConfig
    m = VaswaniForConditionalGeneration(VaswaniConfig())
    m.save_pretrained("data/ckpt"); m2 = VaswaniForConditionalGeneration.from_pretrained("data/ckpt")
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from architecture.configuration_vaswani import VaswaniConfig


# =============================================================================
# Mask helpers. Attention masks here are *additive*: 0.0 keeps a position,
# a large negative number (dtype min) zeroes it out after softmax.
# =============================================================================
def _expand_padding_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    """``[B, S]`` (1=keep) -> additive ``[B, 1, tgt_len, S]`` for broadcasting over heads."""
    bsz, src_len = mask.shape
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded  # 1.0 where we must mask
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


def _make_causal_mask(tgt_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Lower-triangular additive mask ``[tgt_len, tgt_len]``: a token sees itself and the past only."""
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    return torch.triu(mask, diagonal=1)  # strict-upper (future) stays min; diag + below -> 0


# =============================================================================
# Core building blocks
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos positional encoding from the paper (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # persistent=False: deterministic, so we recompute it instead of storing it
        # in the checkpoint (keeps model.safetensors to learned weights only).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, d_model]
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention split across ``n_heads``.

    Handles both self-attention (``key_value_states is None``) and cross-attention
    (queries from the decoder, keys/values from the encoder).
    """

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scaling = self.head_dim**-0.5  # 1/sqrt(d_k), folded into the query
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        # [B, T, d_model] -> [B, n_heads, T, head_dim]
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

        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, Tq, Tk]
        if attention_mask is not None:
            scores = scores + attention_mask  # additive mask broadcasts over heads
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(self.dropout(weights), v)  # [B, H, Tq, head_dim]

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, -1)  # merge heads
        attn = self.out_proj(attn)
        return attn, (weights if output_attentions else None)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> activation -> Linear (paper uses ReLU)."""

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Self-attention + FFN, each wrapped in a post-LN residual block."""

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
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
    """Masked self-attention + cross-attention + FFN, each a post-LN residual block."""

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.cross_attn = MultiHeadAttention(config)
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
# Stacks. Each holds a *reference* to the shared embedding so weights are tied.
# =============================================================================
class VaswaniEncoder(nn.Module):
    def __init__(self, config: VaswaniConfig, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))

    def forward(self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs):
        hidden = self.dropout(self.pos_enc(self.embed_tokens(input_ids) * self.embed_scale))
        mask = None
        if attention_mask is not None:
            mask = _expand_padding_mask(attention_mask, hidden.dtype, input_ids.size(1))

        attentions = () if output_attentions else None
        for layer in self.layers:
            hidden, attn = layer(hidden, mask, output_attentions=output_attentions)
            if output_attentions:
                attentions += (attn,)
        return BaseModelOutput(last_hidden_state=hidden, attentions=attentions)


class VaswaniDecoder(nn.Module):
    def __init__(self, config: VaswaniConfig, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, output_attentions=False):
        tgt_len = input_ids.size(1)
        hidden = self.dropout(self.pos_enc(self.embed_tokens(input_ids) * self.embed_scale))

        # Self-attention mask = causal, plus the decoder's own padding mask if given.
        self_mask = _make_causal_mask(tgt_len, hidden.dtype, hidden.device)[None, None, :, :]
        if attention_mask is not None:
            self_mask = self_mask + _expand_padding_mask(attention_mask, hidden.dtype, tgt_len)
        # Cross-attention mask = the *encoder's* padding mask (queries are decoder positions).
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
# HF-facing models
# =============================================================================
class VaswaniPreTrainedModel(PreTrainedModel):
    """Shared base: wires the config class and weight initialization into HF."""

    config_class = VaswaniConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        # HF convention: normal(0, initializer_range). The 2017 paper used Xavier
        # uniform instead; swap this body for nn.init.xavier_uniform_ to match it.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VaswaniModel(VaswaniPreTrainedModel):
    """Embeddings + encoder + decoder. Returns hidden states (no LM head)."""

    # The encoder and decoder hold references to ``shared``, so their embedding
    # weights are the same tensor; 5.x requires that tying be declared here.
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = VaswaniEncoder(config, self.shared)
        self.decoder = VaswaniDecoder(config, self.shared)
        self.post_init()  # runs _init_weights + ties weights per the config

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.shared = value
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value

    def get_encoder(self) -> VaswaniEncoder:
        return self.encoder

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, encoder_outputs=None, output_attentions=None,
                return_dict=None, **kwargs) -> Seq2SeqModelOutput:
        output_attentions = bool(output_attentions)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                           output_attentions=output_attentions)
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(*encoder_outputs)  # tuple from generation

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


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Build decoder inputs from labels: prepend <bos>, drop the final token.

    So labels ``[w1, w2, <eos>]`` become decoder inputs ``[<bos>, w1, w2]`` — the
    model predicts position t+1 from position t (teacher forcing).
    """
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)  # -100 is the loss-ignore id
    return shifted


class VaswaniForConditionalGeneration(VaswaniPreTrainedModel, GenerationMixin):
    """Full seq2seq model with a tied LM head and ``.generate()`` support."""

    # transformers 5.x: a dict mapping each tied weight to its source weight.
    _tied_weights_keys = {"lm_head.weight": "model.shared.weight"}

    def __init__(self, config: VaswaniConfig) -> None:
        super().__init__(config)
        self.model = VaswaniModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    # --- plumbing HF uses for weight tying and generation --------------------
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
        # No KV cache: feed the full decoder prefix each step and reuse the
        # already-computed encoder_outputs so the encoder runs only once.
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }


if __name__ == "__main__":
    # Smoke test: build, round-trip a checkpoint, run a forward + generate.
    cfg = VaswaniConfig(vocab_size=15, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = VaswaniForConditionalGeneration(cfg).eval()

    src = torch.tensor([[1, 5, 6, 7, 2]])           # <bos> ... <eos>
    labels = torch.tensor([[8, 9, 2]])
    out = model(input_ids=src, labels=labels)
    print(f"loss={out.loss.item():.3f}  logits={tuple(out.logits.shape)}")

    gen = model.generate(input_ids=src, max_new_tokens=5)
    print(f"generated ids: {gen.tolist()}")

    model.save_pretrained("data/ckpt_smoketest")
    reloaded = VaswaniForConditionalGeneration.from_pretrained("data/ckpt_smoketest")
    print(f"reloaded OK, tied lm_head==shared: "
          f"{reloaded.lm_head.weight.data_ptr() == reloaded.get_input_embeddings().weight.data_ptr()}")
