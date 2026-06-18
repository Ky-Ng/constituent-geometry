"""Prompt set for experiment-15: v2 grammar depth-2 random frame dataset.

Parses prompts.md (markdown-table format) and exposes PROMPTS and helpers
used by extract_attention.py and gather_heatmaps.py.
"""

import re
from pathlib import Path

_MD = Path(__file__).resolve().parent / "prompts.md"

_FIELDS = [
    "frame_id", "depth", "tree_height", "hi", "hf",
    "hi_bracketed", "hf_bracketed",
    "hi_frame_tagged", "hf_frame_tagged",
    "is_ambiguous", "n_tokens",
]


def _parse() -> list[dict]:
    prompts = []
    section = None
    row_idx = 0
    with open(_MD) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("## "):
                continue
            if line.startswith("### "):
                section = line[4:].strip()
                row_idx = 0
                continue
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) != 11:
                continue
            if parts[0] in ("frame_id", "---"):
                continue
            row = dict(zip(_FIELDS, parts))
            prompts.append({
                "section":         section,
                "row":             row_idx,
                "hi":              row["hi"],
                "hf":              row["hf"],
                "hi_frame_tagged": row["hi_frame_tagged"],
                "hf_frame_tagged": row["hf_frame_tagged"],
                "depth":           int(row["depth"]),
            })
            row_idx += 1
    return prompts


PROMPTS = _parse()


def prompt_name(section: str, row: int) -> str:
    return f"{section}-{row}"


def build_word_id_map(hi: str, hi_frame_tagged: str) -> dict[str, str]:
    """Map each word in hi to its numeric ID extracted from hi_frame_tagged.

    E.g. hi='John consoles Shri', hi_frame_tagged='[S [DP_1] [VP [V_dp_2] [DP_3]]]'
    -> {'John': '1', 'consoles': '2', 'Shri': '3'}
    Returns an empty dict if the leaf-ID count doesn't match the word count.
    """
    words = hi.split()
    ids = re.findall(r"\b\w+_(\d+)\b", hi_frame_tagged)
    if len(ids) != len(words):
        return {}
    return dict(zip(words, ids))


def label_tokens(tokens: list[str], word_to_id: dict[str, str]) -> list[str]:
    """Attach structural ID suffix to each token: 'John' -> 'John_1'.

    Falls back to the bare token string if the word isn't in word_to_id
    (e.g. BOS/EOS sentinels, or if build_word_id_map returned empty).
    """
    return [f"{t}_{word_to_id[t]}" if t in word_to_id else t for t in tokens]
