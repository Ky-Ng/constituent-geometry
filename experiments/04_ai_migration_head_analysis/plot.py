"""Heatmaps for per-head patching results.

Color encodes polarity of a normalized effect (0 = clean run, 1 = fully counterfactual), so the
scale is diverging: blue (pushed *away* from the counterfactual, < 0) - neutral gray (no effect) -
red (pushed *toward* the counterfactual, > 0). Values outside [-1, 1] are clipped in color but
printed exactly in the cell.
"""
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CJK_FONTS = ["Noto Sans CJK SC", "DejaVu Sans"]
DIVERGING = LinearSegmentedColormap.from_list("effect", ["#184f95", "#f0efec", "#b3261e"])
INK_DARK, INK_LIGHT = "#1a1a19", "#fcfcfb"


def _head_label(layer: int, heads: list[int]) -> str:
    return f"L{layer} H{'+'.join(map(str, heads))}"


def _draw(*, values: np.ndarray, text: list[list[str]], x_labels: list[str], y_labels: list[str],
          x_axis_label: str, title: str, out_path: str | Path) -> None:
    plt.rcParams["font.sans-serif"] = CJK_FONTS
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.15), max(3, len(y_labels) * 0.9 + 1.2)))
    im = ax.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, text[r][c], ha="center", va="center", fontsize=8,
                    color=INK_LIGHT if abs(v) >= 0.55 else INK_DARK)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("layer / head")
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("normalized effect (0 = clean, 1 = counterfactual)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cell_text(row: dict, clean_top_token: str) -> str:
    s = f"{row['normalized_effect']:.2f}"
    if row["top_token"] != clean_top_token:
        s += f"\n→{row['top_token']}"
    return s


def plot_site_heatmap(*, rows: list[dict], site: str, position_labels: list[str], clean_top_token: str,
                      title: str, out_path: str | Path, cell_text: Callable[[dict], str] | None = None) -> None:
    """Rows = single heads (layer, head); columns = patched position, plus ALL. Both-head rows are
    left to `plot_whole_head_summary`. `cell_text(row) -> str`, if given, overrides the default
    "effect + arrow-to-top-token" cell label (used by sweep.py to print mean±std)."""
    single = [r for r in rows if len(r["heads"]) == 1]
    keys = sorted({(r["layer"], r["heads"][0]) for r in single})
    n_pos = len(position_labels)
    values = np.full((len(keys), n_pos + 1), np.nan)
    text = [[""] * (n_pos + 1) for _ in keys]
    for r in single:
        y = keys.index((r["layer"], r["heads"][0]))
        x = n_pos if r["positions"] is None else r["positions"][0]
        values[y, x] = r["normalized_effect"]
        text[y][x] = cell_text(r) if cell_text is not None else _cell_text(r, clean_top_token)
    _draw(values=values, text=text, x_labels=position_labels + ["ALL"],
          y_labels=[_head_label(l, [h]) for l, h in keys],
          x_axis_label=f"patched position (index meaning depends on site: {site})", title=title, out_path=out_path)


def plot_whole_head_summary(rows: list[dict], *, sites: list[str], title: str, out_path: str | Path,
                            cell_text: Callable[[dict], str] | None = None) -> None:
    """Rows = (layer, heads) including both-heads-at-once; columns = site. All positions patched.
    `cell_text(row) -> str`, if given, overrides the default "effect + top token" cell label."""
    keys = sorted({(r["layer"], tuple(r["heads"])) for r in rows}, key=lambda k: (k[0], len(k[1]), k[1]))
    values = np.full((len(keys), len(sites)), np.nan)
    text = [[""] * len(sites) for _ in keys]
    clean_top = None
    for r in rows:
        y = keys.index((r["layer"], tuple(r["heads"])))
        x = sites.index(r["site"])
        values[y, x] = r["normalized_effect"]
        text[y][x] = cell_text(r) if cell_text is not None else f"{r['normalized_effect']:.2f}\n{r['top_token']}"
    _draw(values=values, text=text, x_labels=list(sites), y_labels=[_head_label(l, list(h)) for l, h in keys],
          x_axis_label="patched site (all positions, all-at-once for the listed heads)", title=title, out_path=out_path)
