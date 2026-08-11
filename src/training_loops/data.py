"""Dataset/tokenization for the HI<->HF translation task.

Each row in data/grammar_samples.csv yields two training sequences:
    HI -> HF:  <hi> s_hi <translate> s_hf <eos>
    HF -> HI:  <hf> s_hf <translate> s_hi <eos>
Pad with <pad>. The first token (<hi> or <hf>) signals the translation direction.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from grammar.vocab import BOS, EOS, HF, HI, PAD, TRANSLATE, build_vocab


@dataclass
class EncodedExample:
    tokens: torch.Tensor     # (T,)
    attn_mask: torch.Tensor  # (T,) 1=valid, 0=pad
    loss_mask: torch.Tensor  # (T,) 1 where this position's *target* contributes to loss


def _encode_seq(
    direction_token: str,
    source_tokens: list[str],
    target_tokens: list[str],
    stoi: dict[str, int],
    max_seq_len: int,
    loss_on: str,
) -> EncodedExample:
    seq = [direction_token] + source_tokens + [TRANSLATE] + target_tokens + [EOS]
    if len(seq) > max_seq_len:
        raise ValueError(f"sequence length {len(seq)} > max_seq_len {max_seq_len}")

    ids = [stoi[t] for t in seq]
    pad_len = max_seq_len - len(ids)
    attn = [1] * len(ids) + [0] * pad_len
    ids = ids + [stoi[PAD]] * pad_len

    # loss_mask flags positions whose *contents* count as ground-truth targets.
    # In causal_lm_loss we read loss_mask[:, 1:] — so index i of loss_mask corresponds
    # to token i being predicted from context 0..i-1.
    loss_mask = [0] * max_seq_len
    if loss_on == "all":
        # every non-pad token (including the direction prefix and the whole source/target).
        # The shift in causal_lm_loss handles the first-token / padding edge correctly.
        for i in range(len(seq)):
            loss_mask[i] = 1
    elif loss_on == "target":
        # only target-side tokens + EOS. Source tokens are conditioning, not supervision.
        translate_pos = 1 + len(source_tokens)  # index of <translate>
        # targets start at translate_pos + 1 and run through EOS (inclusive)
        for i in range(translate_pos + 1, len(seq)):
            loss_mask[i] = 1
    else:
        raise ValueError(f"loss_on must be 'all' or 'target', got {loss_on!r}")

    return EncodedExample(
        tokens=torch.tensor(ids, dtype=torch.long),
        attn_mask=torch.tensor(attn, dtype=torch.long),
        loss_mask=torch.tensor(loss_mask, dtype=torch.long),
    )


class GrammarTranslationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        max_seq_len: int = 16,
        directions: Iterable[str] = ("hi2hf", "hf2hi"),
        loss_on: str = "target",
    ):
        self.tokens_list, self.stoi = build_vocab()
        directions = tuple(directions)
        for d in directions:
            if d not in ("hi2hf", "hf2hi"):
                raise ValueError(f"unknown direction {d!r}")

        self.examples: list[EncodedExample] = []
        with Path(csv_path).open() as f:
            for row in csv.DictReader(f):
                hi = row["hi_surface"].split()
                hf = row["hf_surface"].split()
                if "hi2hf" in directions:
                    self.examples.append(_encode_seq(HI, hi, hf, self.stoi, max_seq_len, loss_on))
                if "hf2hi" in directions:
                    self.examples.append(_encode_seq(HF, hf, hi, self.stoi, max_seq_len, loss_on))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {"tokens": ex.tokens, "attn_mask": ex.attn_mask, "loss_mask": ex.loss_mask}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens_list)


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}
