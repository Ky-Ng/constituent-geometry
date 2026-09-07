"""Per-head activation patching from a counterfactual prompt into an original prompt.

Every site is patched *one head at a time*. The meaning of the position index differs by site:

| site    | hook                      | shape                       | index means            |
|---------|---------------------------|-----------------------------|------------------------|
| z       | blocks.L.attn.hook_z      | [batch, pos, head, d_head]  | query (output) position |
| q       | blocks.L.attn.hook_q      | [batch, pos, head, d_head]  | query position          |
| k       | blocks.L.attn.hook_k      | [batch, pos, head, d_head]  | source (key) position   |
| v       | blocks.L.attn.hook_v      | [batch, pos, head, d_head]  | source (value) position |
| pattern | blocks.L.attn.hook_pattern| [batch, head, q_pos, k_pos] | query row               |

`positions=None` means "every position" (the classic whole-head patch).
"""
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

from metrics import logit_diff

SITES: tuple[str, ...] = ("z", "q", "k", "v", "pattern")
_SITE_TO_HOOK = {
    "z": "attn.hook_z",
    "q": "attn.hook_q",
    "k": "attn.hook_k",
    "v": "attn.hook_v",
    "pattern": "attn.hook_pattern",
}


def hook_name(layer: int, site: str) -> str:
    return f"blocks.{layer}.{_SITE_TO_HOOK[site]}"


def make_patch_hook(
    site: str,
    heads: Sequence[int],
    positions: Sequence[int] | None,
    source: Tensor,
) -> Callable[[Tensor, HookPoint], Tensor]:
    """Build a hook that overwrites the chosen heads (and positions) with `source`, in place."""
    pos_index = slice(None) if positions is None else list(positions)

    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        for h in heads:
            if site == "pattern":
                activation[:, h, pos_index, :] = source[:, h, pos_index, :]
            else:
                activation[:, pos_index, h, :] = source[:, pos_index, h, :]
        return activation

    return hook_fn


@torch.inference_mode()
def run_patched(
    model: TransformerBridge,
    original_tokens: Int[Tensor, "batch seq"],
    *,
    layer: int,
    site: str,
    heads: Sequence[int],
    positions: Sequence[int] | None,
    cache_counterfactual: ActivationCache,
) -> Float[Tensor, "batch seq vocab"]:
    name = hook_name(layer, site)
    hook = make_patch_hook(site, heads, positions, cache_counterfactual[name])
    return model.run_with_hooks(original_tokens, fwd_hooks=[(name, hook)])


@dataclass(frozen=True)
class PatchCell:
    """One intervention and its readout at the critical position."""
    site: str
    layer: int
    heads: tuple[int, ...]
    positions: tuple[int, ...] | None  # None = all positions
    logit_diff: float
    top_token: str

    def to_json(self) -> dict:
        d = asdict(self)
        d["positions"] = None if self.positions is None else list(self.positions)
        d["heads"] = list(self.heads)
        return d


@torch.inference_mode()
def sweep_site(
    model: TransformerBridge,
    original_tokens: Int[Tensor, "batch seq"],
    cache_counterfactual: ActivationCache,
    *,
    site: str,
    critical_pos: int,
    answer_id: int,
    counterfactual_id: int,
) -> list[PatchCell]:
    """For one site: every (layer, head) x every single position, plus every (layer, head) x ALL
    positions, plus every layer with BOTH heads x ALL positions."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    seq_len = original_tokens.shape[1]
    cells: list[PatchCell] = []

    def record(layer: int, heads: Sequence[int], positions: Sequence[int] | None) -> None:
        logits = run_patched(
            model, original_tokens,
            layer=layer, site=site, heads=heads, positions=positions,
            cache_counterfactual=cache_counterfactual,
        )
        cells.append(PatchCell(
            site=site, layer=layer, heads=tuple(heads),
            positions=None if positions is None else tuple(positions),
            logit_diff=logit_diff(logits, critical_pos, answer_id, counterfactual_id),
            top_token=model.to_single_str_token(int(logits[0, critical_pos].argmax())),
        ))

    for layer in range(n_layers):
        for head in range(n_heads):
            for pos in range(seq_len):
                record(layer, [head], [pos])
            record(layer, [head], None)
        record(layer, list(range(n_heads)), None)
    return cells
