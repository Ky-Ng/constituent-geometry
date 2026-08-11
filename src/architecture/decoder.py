"""Attention-only decoder-only transformer, GPT-2 style but without FFN.

Faithful to the planning doc spec:
    - Decoder only, causal self-attention
    - Pre-LayerNorm (GPT-2 style)
    - Learned positional embeddings
    - Attention-only blocks (no MLP/FFN) to isolate attention-mediated information flow
    - Untied input embedding and output unembedding (planning: "Do not use tied weights")

Defaults match the toy configuration: n_layers=2, n_heads=1, d_model=3.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 3
    n_layers: int = 2
    n_heads: int = 1
    max_seq_len: int = 16
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.W_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        B, T, C = x.shape
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) with 1 for valid, 0 for pad → mask those key positions
            scores = scores.masked_fill(~key_padding_mask.bool()[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                               # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.W_o(out))
        return out, attn


class Block(nn.Module):
    """Pre-LN attention-only block. No FFN."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

    def forward(self, x, key_padding_mask=None):
        a, attn_weights = self.attn(self.ln(x), key_padding_mask=key_padding_mask)
        return x + a, attn_weights


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """tokens: (B, T) long; key_padding_mask: (B, T) 1=valid, 0=pad."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        attn_all = []
        for blk in self.blocks:
            x, aw = blk(x, key_padding_mask=key_padding_mask)
            attn_all.append(aw)
        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits, attn_all

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def causal_lm_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard next-token CE. `tokens` is the full seq; `logits` are over the full seq.

    logits:   (B, T, V)
    tokens:   (B, T)    long
    loss_mask:(B, T)    1 on positions where the *target* should contribute to loss, else 0.
              The shift (drop last input / first target) is handled here.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tokens[:, 1:].contiguous()
    V = shift_logits.size(-1)
    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_targets.reshape(-1), reduction="none"
    ).view(shift_targets.shape)
    if loss_mask is not None:
        m = loss_mask[:, 1:].float()
        denom = m.sum().clamp_min(1.0)
        return (per_tok * m).sum() / denom
    return per_tok.mean()
