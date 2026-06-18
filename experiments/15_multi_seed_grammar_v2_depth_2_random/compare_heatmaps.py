"""Cross-seed attention-heatmap CONSISTENCY grids for experiment 09.

GOAL
----
For ONE architecture and ONE fixed prompt, show whether the S seeds learned the
same attention pattern. Per attention family, per layer, we render an S x H grid:
    rows    = seeds   (e.g. 42..51)
    columns = heads
So the Vaswani-original model (L=4, H=4, S=10) produces 4 images per family
(one per layer), each a 10 x 4 grid. Encoder-decoder models emit all three
families (encoder_self, decoder_self, cross); gpt2_rope emits one ("self").

HEAD ALIGNMENT (default on)
---------------------------
Attention heads within a layer have NO canonical order: permuting the heads (and
the matching slices of the projection weights) yields a functionally identical
model. So a naive "seed A head i vs seed B head i" comparison compares arbitrary
labels. We instead match each seed's heads to a REFERENCE seed (the first one)
by the assignment that maximizes total cosine similarity (exact brute force --
only H heads), then reorder that seed's columns so column j shows the head that
best matches reference head j. The averaged matched cosine is a permutation-
INVARIANT consistency score. See the experiment README for a worked example.
Disable the visual reordering with --no-align-heads (the metric is unaffected).

WHY A FIXED GOLD (hi, hf) PAIR
------------------------------
Attention shape/labels depend on the token sequence. We teacher-force the SAME
gold (hi, hf) for every seed, so all grids share identical token axes and tensor
shapes [L, H, Tq, Tk] -- otherwise per-seed grids would not be comparable. The
pair is pulled from the dataset by default (or pass --hi/--hf explicitly).

REUSE
-----
Loading + extraction come straight from src/visualization/attention_heatmaps.py
(ADAPTERS) -- no model logic is duplicated here.

OUTPUT
------
Figures: figures/<arch>/heatmaps/[<prompt-name>/]layer_<l>_<suffix>.png
  suffix = "attention" for single-family gpt2_rope, else the family name.
Numbers: results/<arch>/heatmap_consistency.csv -- one row per (family, layer,
  seed) with mean_matched_cosine (order-invariant) vs mean_naive_cosine
  (same-index). The gap between them is the head-permutation effect.

USAGE
-----
    # default: gold pair = test split row 0 of the training dataset
    uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani
    uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani_rope --split test --index 3
    uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani --no-align-heads
    # explicit prompt instead of a dataset row:
    uv run python experiments/09_multi_seed/compare_heatmaps.py --arch gpt2_rope \\
        --hi "the dancer thinks that a cat knows Betty" --hf "<gold hf string>"
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import torch

# Reuse the loader/extractor adapters; the arch keys ARE the model_type keys.
from visualization.attention_heatmaps import ADAPTERS, _slugify

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"

ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def resolve_prompt(args) -> tuple[str, str]:
    """Return the (hi, hf) gold pair, from --hi/--hf or a dataset row."""
    if args.hi is not None and args.hf is not None:
        return args.hi, args.hf
    if args.hi is not None or args.hf is not None:
        sys.exit("Pass BOTH --hi and --hf, or neither (to use a dataset row).")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    if args.index >= len(ds):
        sys.exit(f"--index {args.index} out of range for split '{args.split}' (len {len(ds)}).")
    row = ds[args.index]
    return row[args.src], row[args.tgt]


def _file_suffix(family: str) -> str:
    """Single-family gpt2_rope is named '..._attention' per the requested layout."""
    return "attention" if family == "self" else family


# ============================================================================
# Head matching (the assignment problem). The optimal pairing IS the column
# reordering; its averaged matched cosine IS the order-invariant similarity.
# ============================================================================
def _cosine_matrix(ref: torch.Tensor, oth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ref [H, D], oth [H, D] -> [H, H] cosine sims (row i = reference head i)."""
    r = ref / (ref.norm(dim=1, keepdim=True) + eps)
    o = oth / (oth.norm(dim=1, keepdim=True) + eps)
    return r @ o.t()


def _best_assignment(sim: torch.Tensor) -> tuple[list[int], list[float]]:
    """Maximize total cosine under a one-to-one head matching.

    `sim` is [H, H] (reference heads as rows, other-seed heads as cols). Returns
    (perm, matched) where perm[i] is the other-seed head assigned to reference
    head i and matched[i] = sim[i, perm[i]]. Exact via brute force for H <= 8
    (H is tiny -- 4 here), greedy fallback beyond.
    """
    H = sim.shape[0]
    s = sim.tolist()
    if H <= 8:
        best_total, best_perm = None, None
        for perm in itertools.permutations(range(H)):
            total = sum(s[i][perm[i]] for i in range(H))
            if best_total is None or total > best_total:
                best_total, best_perm = total, perm
        perm = list(best_perm)
    else:
        # Greedy: repeatedly take the largest remaining (row, col) pair.
        perm = [-1] * H
        used_rows, used_cols = set(), set()
        for _, i, j in sorted(((s[i][j], i, j) for i in range(H) for j in range(H)),
                              reverse=True):
            if i in used_rows or j in used_cols:
                continue
            perm[i] = j
            used_rows.add(i)
            used_cols.add(j)
            if len(used_rows) == H:
                break
    matched = [s[i][perm[i]] for i in range(H)]
    return perm, matched


def align_heads_to_reference(vecs, ref_idx):
    """Match every seed's heads to the reference seed for ONE (family, layer).

    `vecs` is a list (one per seed, same order as `seeds`) of [H, D] matrices --
    each row is one head's attention map flattened to a vector. Returns three
    seed-indexed lists:
      perms[s][j]   = the seed-s head matched to REFERENCE head j (identity for ref)
      matched[s][j] = cosine of that matched pair (1.0 for the reference seed)
      naive[s]      = mean of the diagonal (same-index head i vs head i) cosines
    `perms`/`matched` drive the column reordering + per-cell tags; the gap between
    the mean of `matched` and `naive` is exactly the head-permutation effect.

    Single source of truth for the head-matching metric: both the single-prompt
    tool (main) and the multi-prompt gather step (gather_heatmaps.py) call this,
    as will the planned similarity-metric iteration.
    """
    n_seeds = len(vecs)
    n_heads = vecs[ref_idx].shape[0]
    ref_vecs = vecs[ref_idx]
    perms, matched, naive = [], [], []
    for s in range(n_seeds):
        if s == ref_idx:
            perms.append(list(range(n_heads)))
            matched.append([1.0] * n_heads)
            naive.append(1.0)
            continue
        sim = _cosine_matrix(ref_vecs, vecs[s])
        perm, m = _best_assignment(sim)
        perms.append(perm)
        matched.append(m)
        naive.append(float(sim.diag().mean()))
    return perms, matched, naive


def plot_grid(per_seed_attn, seeds, layer, qtok, ktok, title, save_path,
              perms, matched, ref_idx, align):
    """One image: S rows (seeds) x H cols (heads) of square heatmaps for `layer`.

    When `align`, column j shows each seed's head perms[s][j] (the head matched
    to reference head j); otherwise raw head j. Each cell is annotated with its
    original head index and (for aligned non-reference rows) the matched cosine.
    """
    import matplotlib.pyplot as plt

    n_seeds = len(seeds)
    n_heads = per_seed_attn[0].shape[1]                  # [L, H, Tq, Tk]
    # Every cell carries token labels on both axes (below). Rather than shrink
    # the font, we grow the canvas so each cell is large enough for readable
    # words even for the longest (depth-3) sequences.
    fig, axes = plt.subplots(
        n_seeds, n_heads,
        figsize=(3.6 * n_heads + 1.0, 3.6 * n_seeds + 1.0),
        squeeze=False,
    )
    for si, seed in enumerate(seeds):
        attn = per_seed_attn[si]                          # [L, H, Tq, Tk]
        for j in range(n_heads):
            orig = perms[si][j] if align else j
            ax = axes[si][j]
            ax.imshow(attn[layer, orig], cmap="viridis", vmin=0.0, vmax=1.0,
                      aspect="auto")
            # Force a SQUARE rendered box regardless of Tq vs Tk (cross-attn has
            # Tq != Tk), which is what "square heatmaps" means here.
            ax.set_box_aspect(1)
            if si == 0:
                ax.set_title(f"ref h{j}" if align else f"head {j}", fontsize=9)
            if j == 0:
                lbl = f"seed {seed}" + ("  (ref)" if si == ref_idx else "")
                ax.set_ylabel(lbl, fontsize=9)
            # Per-cell tag: which original head this is (+ matched cosine when
            # aligned and not the reference row).
            if align and si != ref_idx:
                tag = f"h{orig}·{matched[si][j]:.2f}"
            else:
                tag = f"h{orig}"
            ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=6,
                    va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.45))
            # Token words on BOTH axes of EVERY cell, so each heatmap is
            # self-contained (readable on its own when cropped/zoomed) rather
            # than only along the grid's outer edge. All cells share the same
            # token axes (one fixed prompt), so these labels repeat by design;
            # font is small to keep the long depth-3 sequences from overlapping.
            ax.set_xticks(range(len(ktok)))
            ax.set_xticklabels(ktok, rotation=90, fontsize=7)
            ax.set_yticks(range(len(qtok)))
            ax.set_yticklabels(qtok, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=ARCHS, required=True,
                   help="Which experiment's seeds to compare (also the model_type).")
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51.")
    p.add_argument("--align-heads", action=argparse.BooleanOptionalAction, default=True,
                   help="Reorder each seed's head columns to best-match the reference "
                        "seed (per family, layer). --no-align-heads keeps raw head "
                        "order; the order-invariant metric is computed either way.")
    p.add_argument("--reference-seed", type=int, default=None,
                   help="Seed to align others against (must be in --seeds). "
                        "Default: the first seed.")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Only used if a gold --hf is not resolved (not the default path).")
    # Prompt source: a dataset row (default) OR an explicit --hi/--hf pair.
    p.add_argument("--dataset", default="kylelovesllms/hi_hf_frames_d3_random_100")
    p.add_argument("--split", default="test")
    p.add_argument("--index", type=int, default=0, help="Row index within --split.")
    p.add_argument("--src", default="hi")
    p.add_argument("--tgt", default="hf")
    p.add_argument("--hi", default=None, help="Override: HI prompt (requires --hf).")
    p.add_argument("--hf", default=None, help="Override: gold HF target (requires --hi).")
    p.add_argument("--prompt-name", default=None,
                   help="Optional subfolder under heatmaps/ so multiple prompts "
                        "don't clobber each other.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    ref_idx = 0
    if args.reference_seed is not None:
        if args.reference_seed not in seeds:
            sys.exit(f"--reference-seed {args.reference_seed} not in --seeds {seeds}.")
        ref_idx = seeds.index(args.reference_seed)

    hi, hf = resolve_prompt(args)
    load, extract = ADAPTERS[args.arch]["load"], ADAPTERS[args.arch]["extract"]

    print(f"arch={args.arch}  seeds={seeds}  reference=seed {seeds[ref_idx]}  "
          f"align={args.align_heads}\n  HI: {hi!r}\n  HF: {hf!r}")

    # --- extract one bundle per seed (load -> extract -> free model) ----------
    bundles = []
    for seed in seeds:
        ckpt = RESULTS_DIR / args.arch / f"seed_{seed}" / "best"
        if not ckpt.exists():
            sys.exit(f"missing checkpoint: {ckpt} (run the seed sweep first)")
        model, tok = load(str(ckpt), args.device)
        bundles.append(extract(model, tok, hi, hf, args.device, args.max_new_tokens))
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"  extracted seed {seed}")

    families = list(bundles[0].groups.keys())             # 3 for enc-dec, 1 for gpt2
    n_layers = bundles[0].groups[families[0]].shape[0]
    n_seeds = len(seeds)

    # All seeds must share the same shape (guaranteed by the fixed gold pair).
    for family in families:
        ref_shape = bundles[0].groups[family].shape
        for seed, b in zip(seeds, bundles):
            if b.groups[family].shape != ref_shape:
                sys.exit(f"shape mismatch for seed {seed}, family '{family}': "
                         f"{tuple(b.groups[family].shape)} != {tuple(ref_shape)}. "
                         f"The gold (hi, hf) must tokenize identically across seeds.")

    slug = _slugify(hi)
    out_root = FIGURES_DIR / args.arch / "heatmaps"
    if args.prompt_name:
        out_root = out_root / args.prompt_name
    out_root.mkdir(parents=True, exist_ok=True)

    csv_rows: list[tuple] = []
    summary: dict[str, list[tuple[float, float]]] = {fam: [] for fam in families}
    nonref = [si for si in range(n_seeds) if si != ref_idx]
    written = 0

    # --- one image (+ metrics) per (family, layer) ---------------------------
    for family in families:
        per_seed = [b.groups[family] for b in bundles]    # list of [L, H, Tq, Tk]
        n_heads = per_seed[0].shape[1]
        qtok = bundles[0].query_tokens[family]
        ktok = bundles[0].key_tokens[family]
        for layer in range(n_layers):
            # Per-seed head-vector matrix [H, Tq*Tk] at this (family, layer).
            vecs = [per_seed[s][layer].reshape(n_heads, -1) for s in range(n_seeds)]
            perms, matched, naive = align_heads_to_reference(vecs, ref_idx)

            # order-invariant (matched) vs naive (same-index), over non-ref seeds.
            oi = sum(sum(matched[s]) / n_heads for s in nonref) / max(len(nonref), 1)
            nv = sum(naive[s] for s in nonref) / max(len(nonref), 1)
            summary[family].append((oi, nv))
            for s in range(n_seeds):
                csv_rows.append((family, layer, seeds[s], seeds[ref_idx],
                                 sum(matched[s]) / n_heads, naive[s]))

            save_path = out_root / f"layer_{layer}_{_file_suffix(family)}.png"
            title = (f"{args.arch} | {family} | layer {layer} | rows=seeds cols=heads"
                     f" | order-inv cos={oi:.3f} (naive {nv:.3f})")
            plot_grid(per_seed, seeds, layer, qtok, ktok, title, save_path,
                      perms, matched, ref_idx, args.align_heads)
            written += 1
            print(f"  wrote {save_path.relative_to(EXP_DIR)}  "
                  f"(order-inv={oi:.3f}, naive={nv:.3f})")

    # --- write CSV + print per-family summary --------------------------------
    csv_path = RESULTS_DIR / args.arch / "heatmap_consistency.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "layer", "seed", "ref_seed",
                    "mean_matched_cosine", "mean_naive_cosine"])
        for family, layer, seed, ref_seed, mm, mn in csv_rows:
            w.writerow([family, layer, seed, ref_seed, f"{mm:.4f}", f"{mn:.4f}"])

    print("\n=== order-invariant consistency (mean over non-ref seeds & layers) ===")
    for family in families:
        vals = summary[family]
        oi = sum(v[0] for v in vals) / len(vals)
        nv = sum(v[1] for v in vals) / len(vals)
        print(f"  {family:<14} order-inv={oi:.3f}  naive={nv:.3f}  (gap={oi - nv:+.3f})")
    print(f"\ndone: {written} image(s); metrics -> {csv_path.relative_to(EXP_DIR)} "
          f"(slug={slug}).")


if __name__ == "__main__":
    main()
