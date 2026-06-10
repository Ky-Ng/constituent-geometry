"""Stage 2 of the experiment-09 cross-seed comparison: GATHER caches -> grids.

Reads the attention bundles cached by extract_attention.py (Stage 1) and, for each
(arch, prompt), renders the S x H consistency grid -- rows = seeds, columns =
heads, columns reference-aligned by the cosine assignment -- one image per
(attention family, layer). No models are loaded here; this is a cheap CPU pass, so
it runs as a single job over all archs x prompts.

OUTPUT (matches the requested layout)
-------------------------------------
Figures: figures/<arch>/heatmaps/depth<D>-ex<N>/layer_<L>_<type>.png
  <type> = "attention" for single-family gpt2_rope, else the family name
  (encoder_self | decoder_self | cross).
Numbers: results/<arch>/heatmap_consistency.csv -- one row per
  (depth, ex, family, layer, seed) with mean_matched_cosine (order-invariant) vs
  mean_naive_cosine (same-index). Written ONCE per arch (all prompts together), so
  prompts never clobber each other. This CSV is what the planned similarity-metric
  iteration will consume / extend.

HEAD ALIGNMENT and the metric are identical to compare_heatmaps.py -- in fact the
matching itself (`align_heads_to_reference`) and the renderer (`plot_grid`) are
IMPORTED from it, so there is one source of truth. See the experiment README and
compare_heatmaps.py's docstring for the worked example.

HOW TO RUN
----------
* All archs (after Stage 1 has populated every cache):
      uv run python experiments/09_multi_seed/gather_heatmaps.py
  or as a (single) CPU job:
      sbatch slurm/run_cpu.sbatch experiments/09_multi_seed/gather_heatmaps.py

* One arch, or raw (non-reordered) columns for a sanity check:
      uv run python experiments/09_multi_seed/gather_heatmaps.py --arch vaswani
      uv run python experiments/09_multi_seed/gather_heatmaps.py --no-align-heads
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

# Single source of truth for the matching + rendering (see compare_heatmaps.py).
from compare_heatmaps import (
    align_heads_to_reference,
    plot_grid,
    _file_suffix,
    parse_seeds,
)
from prompts import PROMPTS, prompt_name

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]


def cache_path(arch: str, seed: int, depth: int, ex: int) -> Path:
    return RESULTS_DIR / arch / f"seed_{seed}" / "attn" / f"{prompt_name(depth, ex)}.pt"


def load_bundles(arch: str, seeds: list[int], depth: int, ex: int) -> list[dict]:
    """Load every seed's cached bundle for one prompt; abort if any is missing."""
    bundles = []
    for seed in seeds:
        path = cache_path(arch, seed, depth, ex)
        if not path.exists():
            sys.exit(f"missing cache: {path}\n  run Stage 1 first: "
                     f"sbatch --array=0-{len(seeds) * len(ARCHS) - 1} "
                     f"slurm/run_cpu.sbatch experiments/09_multi_seed/extract_attention.py")
        bundles.append(torch.load(path, weights_only=False))
    return bundles


def render_prompt(arch, seeds, ref_idx, align, depth, ex) -> list[tuple]:
    """Render all (family, layer) grids for ONE prompt; return its CSV metric rows."""
    bundles = load_bundles(arch, seeds, depth, ex)
    n_seeds = len(seeds)
    families = list(bundles[0]["groups"].keys())            # 3 for enc-dec, 1 for gpt2
    n_layers = bundles[0]["groups"][families[0]].shape[0]

    # All seeds must share each family's shape (guaranteed by the fixed gold pair).
    for family in families:
        ref_shape = bundles[0]["groups"][family].shape
        for seed, b in zip(seeds, bundles):
            if b["groups"][family].shape != ref_shape:
                sys.exit(f"shape mismatch {arch} {prompt_name(depth, ex)} seed {seed} "
                         f"family '{family}': {tuple(b['groups'][family].shape)} != "
                         f"{tuple(ref_shape)} (tokenization must match across seeds).")

    out_root = FIGURES_DIR / arch / "heatmaps" / prompt_name(depth, ex)
    out_root.mkdir(parents=True, exist_ok=True)
    nonref = [si for si in range(n_seeds) if si != ref_idx]
    rows: list[tuple] = []

    for family in families:
        per_seed = [b["groups"][family] for b in bundles]   # list of [L, H, Tq, Tk]
        n_heads = per_seed[0].shape[1]
        qtok = bundles[0]["query_tokens"][family]
        ktok = bundles[0]["key_tokens"][family]
        for layer in range(n_layers):
            # Per-seed head-vector matrix [H, Tq*Tk]; match heads to the reference.
            vecs = [per_seed[s][layer].reshape(n_heads, -1) for s in range(n_seeds)]
            perms, matched, naive = align_heads_to_reference(vecs, ref_idx)

            oi = sum(sum(matched[s]) / n_heads for s in nonref) / max(len(nonref), 1)
            nv = sum(naive[s] for s in nonref) / max(len(nonref), 1)
            for s in range(n_seeds):
                rows.append((depth, ex, family, layer, seeds[s], seeds[ref_idx],
                             sum(matched[s]) / n_heads, naive[s]))

            save_path = out_root / f"layer_{layer}_{_file_suffix(family)}.png"
            title = (f"{arch} | depth{depth}-ex{ex} | {family} | layer {layer} | "
                     f"rows=seeds cols=heads | order-inv cos={oi:.3f} (naive {nv:.3f})")
            plot_grid(per_seed, seeds, layer, qtok, ktok, title, save_path,
                      perms, matched, ref_idx, align)
            print(f"  wrote {save_path.relative_to(EXP_DIR)}  "
                  f"(order-inv={oi:.3f}, naive={nv:.3f})", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=ARCHS, default=None,
                   help="Restrict to one arch. Default: all three.")
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51.")
    p.add_argument("--align-heads", action=argparse.BooleanOptionalAction, default=True,
                   help="Reorder each seed's head columns to best-match the reference "
                        "seed. --no-align-heads keeps raw order; the metric is unaffected.")
    p.add_argument("--reference-seed", type=int, default=None,
                   help="Seed to align others against (must be in --seeds). Default: first.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    ref_idx = 0
    if args.reference_seed is not None:
        if args.reference_seed not in seeds:
            sys.exit(f"--reference-seed {args.reference_seed} not in --seeds {seeds}.")
        ref_idx = seeds.index(args.reference_seed)

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        print(f"\n=== {arch}  seeds={seeds}  reference=seed {seeds[ref_idx]}  "
              f"align={args.align_heads} ===", flush=True)
        arch_rows: list[tuple] = []
        for prompt in PROMPTS:
            print(f"[{prompt_name(prompt['depth'], prompt['ex'])}] {prompt['hi']!r}", flush=True)
            arch_rows.extend(
                render_prompt(arch, seeds, ref_idx, args.align_heads,
                              prompt["depth"], prompt["ex"])
            )

        csv_path = RESULTS_DIR / arch / "heatmap_consistency.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["depth", "ex", "family", "layer", "seed", "ref_seed",
                        "mean_matched_cosine", "mean_naive_cosine"])
            for depth, ex, family, layer, seed, ref_seed, mm, mn in arch_rows:
                w.writerow([depth, ex, family, layer, seed, ref_seed,
                            f"{mm:.4f}", f"{mn:.4f}"])
        print(f"  metrics -> {csv_path.relative_to(EXP_DIR)} ({len(arch_rows)} rows)",
              flush=True)


if __name__ == "__main__":
    main()
