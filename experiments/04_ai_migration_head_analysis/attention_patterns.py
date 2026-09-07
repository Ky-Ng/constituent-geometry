"""Visualize every head's attention pattern for an (original, counterfactual) pair.

One figure per layer: n_heads rows x 3 columns = original pattern, counterfactual pattern, and their
difference (original - counterfactual). Rows of each matrix are query positions, columns are key
positions; the causal mask leaves the upper triangle blank. The critical query row is outlined.
Also prints and saves (JSON) the critical row of every head, which is where the README's
"attention rows" table comes from.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from transformer_lens.model_bridge import TransformerBridge

from plot import CJK_FONTS, DIVERGING, INK_DARK, INK_LIGHT

SEQUENTIAL = LinearSegmentedColormap.from_list("attn", ["#f0efec", "#184f95"])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", default="kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift")
parser.add_argument("--original", required=True)
parser.add_argument("--counterfactual", required=True)
parser.add_argument("--critical_pos", type=int, default=None, help="Default: index of <sep> + 1")
parser.add_argument("--out_dir", required=True, help="Directory for attention_L<layer>.png and attention_rows.json")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


def _draw(ax, mat: np.ndarray, labels_q: list[str], labels_k: list[str], *, title: str, diverging: bool) -> None:
    n = mat.shape[0]
    masked = mat.astype(float).copy()
    masked[np.triu_indices(n, k=1)] = np.nan  # causal mask
    if diverging:
        im = ax.imshow(masked, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    else:
        im = ax.imshow(masked, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal")
    for q in range(n):
        for k in range(q + 1):
            v = masked[q, k]
            if abs(v) < 0.05:
                continue
            ax.text(k, q, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color=INK_LIGHT if abs(v) >= 0.55 else INK_DARK)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_k, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_q, fontsize=7)
    ax.set_xlabel("key (source) position", fontsize=8)
    ax.set_ylabel("query position", fontsize=8)
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


@torch.inference_mode()
def main() -> None:
    plt.rcParams["font.sans-serif"] = CJK_FONTS
    plt.rcParams["axes.unicode_minus"] = False
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device)
    tok_o = model.to_tokens(args.original, prepend_bos=False)
    tok_c = model.to_tokens(args.counterfactual, prepend_bos=False)
    str_o, str_c = model.to_str_tokens(tok_o), model.to_str_tokens(tok_c)
    assert len(str_o) == len(str_c), (str_o, str_c)
    critical = args.critical_pos if args.critical_pos is not None else str_o.index("<sep>") + 1
    n = len(str_o)
    lab_o = [f"{i}_{t}" for i, t in enumerate(str_o)]
    lab_c = [f"{i}_{t}" for i, t in enumerate(str_c)]

    pat = lambda name: name.endswith("hook_pattern")  # noqa: E731
    _, cache_o = model.run_with_cache(tok_o, names_filter=pat)
    _, cache_c = model.run_with_cache(tok_c, names_filter=pat)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    for L in range(model.cfg.n_layers):
        P_o = cache_o[f"blocks.{L}.attn.hook_pattern"][0].cpu().numpy()  # [head, q, k]
        P_c = cache_c[f"blocks.{L}.attn.hook_pattern"][0].cpu().numpy()
        H = P_o.shape[0]
        fig, axes = plt.subplots(H, 3, figsize=(3 * (0.42 * n + 1.5), H * (0.42 * n + 1.2)))
        axes = np.atleast_2d(axes)
        for h in range(H):
            _draw(axes[h, 0], P_o[h], lab_o, lab_o, title=f"L{L} H{h} original", diverging=False)
            _draw(axes[h, 1], P_c[h], lab_c, lab_c, title=f"L{L} H{h} counterfactual", diverging=False)
            _draw(axes[h, 2], P_o[h] - P_c[h], lab_o, lab_o, title=f"L{L} H{h} original − counterfactual", diverging=True)
            for ax in axes[h]:
                ax.add_patch(Rectangle((-0.5, critical - 0.5), n, 1, fill=False, lw=1.5, ec="#1a1a19"))
            for tag, P, labs in (("original", P_o, str_o), ("counterfactual", P_c, str_c)):
                row = P[h, critical]
                rows[f"L{L}H{h}@{critical}_{tag}"] = {f"{i}_{labs[i]}": round(float(row[i]), 3) for i in range(n)}
                top = np.argsort(-row)[:4]
                print(f"{tag:14s} L{L}H{h} query {critical} ({labs[critical]}): " +
                      ", ".join(f"{i}_{labs[i]}={row[i]:.2f}" for i in top))
        fig.suptitle(f"attention patterns, layer {L} (outlined row = critical query position {critical})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"attention_L{L}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    (out / "attention_rows.json").write_text(json.dumps({"original": str_o, "counterfactual": str_c,
                                                         "critical_pos": critical, "rows": rows}, indent=2))
    print("figures ->", out)


if __name__ == "__main__":
    main()
