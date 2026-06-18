"""Plot when each (arch, seed) model first crossed eval_exact_match >= 0.98.

Reads the 30 per-seed training logs from experiment 15's v2 run (the early-
stopping batch) and draws a horizontal bar chart:
  - x-axis: epoch
  - y-axis: 30 bars, one per (arch, seed), grouped and colored by architecture
  - bar length: the epoch at which that run's eval exact-match first hit 0.98
    (i.e. the trigger point for ThresholdStopCallback, NOT the +1-epoch stop)

A dashed line marks each architecture's mean crossing epoch so the
sinusoidal-vs-RoPE convergence gap is visible at a glance.

Output: figures/architecture_vs_convergence.png

To copy: rename to plot_convergence.py.
Run:  uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/plot_convergence.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
LOG_DIR = EXP_DIR / "logs"
FIG_DIR = EXP_DIR / "figures"
PREFIX = "15_multi_seed_grammar_v2_depth_2_random_v2"

ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]
SEEDS = list(range(42, 52))
THRESHOLD = 0.98
COLORS = {"vaswani": "#4C72B0", "vaswani_rope": "#DD8452", "gpt2_rope": "#55A868"}

# In the printed eval dict, 'eval_exact_match' comes before 'epoch'.
EVAL_RE = re.compile(r"'eval_exact_match': '([0-9.]+)'.*?'epoch': '([0-9.]+)'")


def first_crossing_epoch(log_path: Path, threshold: float) -> float | None:
    """Return the epoch of the first eval whose exact_match >= threshold."""
    if not log_path.exists():
        return None
    with log_path.open() as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m and float(m.group(1)) >= threshold:
                return float(m.group(2))
    return None


def main() -> None:
    # Collect crossing epochs, grouped by architecture.
    crossings: dict[str, list[tuple[int, float | None]]] = {}
    for arch in ARCHS:
        rows = []
        for seed in SEEDS:
            ep = first_crossing_epoch(LOG_DIR / f"{PREFIX}_{arch}_seed_{seed}.log", THRESHOLD)
            rows.append((seed, ep))
            print(f"{arch:14s} seed {seed}: crossed 0.98 at epoch {ep}")
        crossings[arch] = rows

    # Lay out 30 bars top-to-bottom, with a 1-row gap between arch groups.
    y_labels: list[str] = []
    y_pos: list[float] = []
    lengths: list[float] = []
    bar_colors: list[str] = []
    group_spans: dict[str, tuple[float, float, float]] = {}  # arch -> (y_lo, y_hi, mean)

    y = 0.0
    for arch in ARCHS:
        ys_this_group = []
        vals = [ep for _, ep in crossings[arch] if ep is not None]
        for seed, ep in crossings[arch]:
            y_pos.append(y)
            ys_this_group.append(y)
            y_labels.append(f"seed {seed}")
            lengths.append(ep if ep is not None else 0.0)
            bar_colors.append(COLORS[arch])
            y += 1.0
        mean = sum(vals) / len(vals) if vals else 0.0
        group_spans[arch] = (min(ys_this_group), max(ys_this_group), mean)
        y += 2.0  # gap between groups (leaves room for the group mean label)

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(y_pos, lengths, color=bar_colors, height=0.8, zorder=3)

    # Annotate each bar with its epoch value.
    for yp, ln in zip(y_pos, lengths):
        ax.text(ln + 0.01, yp, f"{ln:.2f}", va="center", ha="left", fontsize=7, color="#333")

    # Per-architecture mean crossing line + group label (in the gap below group).
    for arch, (lo, hi, mean) in group_spans.items():
        ax.plot([mean, mean], [lo - 0.5, hi + 0.5], ls="--", lw=1.5,
                color=COLORS[arch], zorder=4)
        ax.text(mean, hi + 1.3, f"{arch}  (mean={mean:.2f})", ha="center", va="center",
                fontsize=9, color=COLORS[arch], fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()  # first arch (vaswani) on top
    ax.set_xlabel("Epoch at which eval exact-match first reached 0.98")
    ax.set_title("Architecture vs. convergence: epoch to hit 0.98 exact-match\n"
                 "(experiment 15, v2 grammar depth-2, 10 seeds/arch)")
    ax.set_xlim(0, max(lengths) * 1.18)
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in ARCHS]
    ax.legend(handles, ARCHS, loc="lower right", fontsize=8, framealpha=0.9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "architecture_vs_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
