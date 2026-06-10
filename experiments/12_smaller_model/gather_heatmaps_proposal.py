"""Stage 2 for experiment 12: GATHER caches -> grids (the 09 pipeline, re-pathed).

WHAT THIS IS
------------
A near-verbatim copy of `experiments/09_multi_seed/gather_heatmaps.py`. It reads the
bundles cached by Stage 1 and renders, per (arch, shape, prompt), the S x H
cross-seed consistency grid (rows = seeds, columns = reference-aligned heads), one
image per (attention family, layer). No models are loaded -- a cheap CPU pass.

The TWO differences from 09:
  1. Path layout threads the `L<l>_H<h>` SHAPE level (see extract_attention.py).
  2. Figures go to the requested experiment-12 layout, keyed by LAYER COUNT:
         figures/<arch>_layer_<l>/heatmaps/depth<D>-ex<N>/layer_<L>_<type>.png
     where `<l>` is the model's n_layers (2 or 3), parsed from the shape `L<l>_H<h>`.
     So `vaswani` L2_H2 -> figures/vaswani_layer_2/heatmaps/..., L3_H2 ->
     figures/vaswani_layer_3/heatmaps/...

The MATCHING math (`align_heads_to_reference`) and the RENDERER (`plot_grid`), plus
`_file_suffix` and the prompt set, are IMPORTED from experiment 09 -- one source of
truth; this file only feeds them cached tensors and writes to 12's paths.

OUTPUT
------
Figures: figures/<arch>_layer_<l>/heatmaps/depth<D>-ex<N>/layer_<L>_<type>.png
  <type> = "attention" for single-family gpt2_rope, else the family name.
Numbers: results/<arch>/<shape>/heatmap_consistency.csv -- one row per
  (depth, ex, family, layer, seed) with mean_matched_cosine vs mean_naive_cosine.

HOW TO RUN
----------
* All archs x both shapes (after Stage 1 has populated every cache):
      uv run python experiments/12_smaller_model/gather_heatmaps.py
  or as a (single) CPU job:
      sbatch slurm/run_cpu.sbatch experiments/12_smaller_model/gather_heatmaps.py

* One arch / one shape / raw (non-reordered) columns for a sanity check:
      uv run python experiments/12_smaller_model/gather_heatmaps.py --arch vaswani
      uv run python experiments/12_smaller_model/gather_heatmaps.py --shapes L2_H2
      uv run python experiments/12_smaller_model/gather_heatmaps.py --no-align-heads
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
NINE_DIR = EXP_DIR.parent / "09_multi_seed"      # single source of truth for matching/rendering
sys.path.insert(0, str(NINE_DIR))

from compare_heatmaps import (                   # noqa: E402  (from 09 dir)
    align_heads_to_reference,
    plot_grid,
    _file_suffix,
    parse_seeds,
)
from prompts import PROMPTS, prompt_name         # noqa: E402  (from 09 dir)

RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]
SHAPES = ["L2_H2", "L3_H2"]


def parse_shapes(spec: str) -> list[str]:
    """'L2_H2,L3_H2' -> ['L2_H2', 'L3_H2'] (comma list, order preserved)."""
    return [s.strip() for s in spec.split(",") if s.strip()]


def shape_n_layers(shape: str) -> int:
    """'L2_H2' -> 2. The figure dir is keyed by this layer count per the spec."""
    return int(shape.split("_")[0][1:])


def cache_path(arch: str, shape: str, seed: int, depth: int, ex: int) -> Path:
    return RESULTS_DIR / arch / shape / f"seed_{seed}" / "attn" / f"{prompt_name(depth, ex)}.pt"


def load_bundles(arch: str, shape: str, seeds: list[int], depth: int, ex: int) -> list[dict]:
    """Load every seed's cached bundle for one prompt; abort if any is missing."""
    bundles = []
    for seed in seeds:
        path = cache_path(arch, shape, seed, depth, ex)
        if not path.exists():
            sys.exit(f"missing cache: {path}\n  run Stage 1 first: "
                     f"sbatch --array=0-{len(seeds) * len(ARCHS) * len(SHAPES) - 1} "
                     f"slurm/run_cpu.sbatch experiments/12_smaller_model/extract_attention.py")
        bundles.append(torch.load(path, weights_only=False))
    return bundles


def render_prompt(arch, shape, seeds, ref_idx, align, depth, ex) -> list[tuple]:
    """Render all (family, layer) grids for ONE prompt; return its CSV metric rows."""
    bundles = load_bundles(arch, shape, seeds, depth, ex)
    n_seeds = len(seeds)
    families = list(bundles[0]["groups"].keys())            # 3 for enc-dec, 1 for gpt2
    n_layers = bundles[0]["groups"][families[0]].shape[0]

    # All seeds must share each family's shape (guaranteed by the fixed gold pair).
    for family in families:
        ref_shape = bundles[0]["groups"][family].shape
        for seed, b in zip(seeds, bundles):
            if b["groups"][family].shape != ref_shape:
                sys.exit(f"shape mismatch {arch} {shape} {prompt_name(depth, ex)} seed {seed} "
                         f"family '{family}': {tuple(b['groups'][family].shape)} != "
                         f"{tuple(ref_shape)} (tokenization must match across seeds).")

    # figures/<arch>_layer_<l>/heatmaps/depth<D>-ex<N>/  (keyed by layer count)
    out_root = (FIGURES_DIR / f"{arch}_layer_{shape_n_layers(shape)}"
                / "heatmaps" / prompt_name(depth, ex))
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
            title = (f"{arch} | {shape} | depth{depth}-ex{ex} | {family} | layer {layer} | "
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
    p.add_argument("--shapes", default="L2_H2,L3_H2",
                   help="Comma list of shape dirs to render. Default both 12 shapes.")
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51.")
    p.add_argument("--align-heads", action=argparse.BooleanOptionalAction, default=True,
                   help="Reorder each seed's head columns to best-match the reference "
                        "seed. --no-align-heads keeps raw order; the metric is unaffected.")
    p.add_argument("--reference-seed", type=int, default=None,
                   help="Seed to align others against (must be in --seeds). Default: first.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    shapes = parse_shapes(args.shapes)
    ref_idx = 0
    if args.reference_seed is not None:
        if args.reference_seed not in seeds:
            sys.exit(f"--reference-seed {args.reference_seed} not in --seeds {seeds}.")
        ref_idx = seeds.index(args.reference_seed)

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        for shape in shapes:
            print(f"\n=== {arch} {shape}  seeds={seeds}  reference=seed {seeds[ref_idx]}  "
                  f"align={args.align_heads} ===", flush=True)
            arch_rows: list[tuple] = []
            for prompt in PROMPTS:
                print(f"[{prompt_name(prompt['depth'], prompt['ex'])}] {prompt['hi']!r}",
                      flush=True)
                arch_rows.extend(
                    render_prompt(arch, shape, seeds, ref_idx, args.align_heads,
                                  prompt["depth"], prompt["ex"])
                )

            csv_path = RESULTS_DIR / arch / shape / "heatmap_consistency.csv"
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
