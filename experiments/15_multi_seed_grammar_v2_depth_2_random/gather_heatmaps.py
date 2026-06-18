"""Stage 2 of the experiment-15 cross-seed comparison: GATHER caches -> grids.

Mirrors experiments/09_multi_seed/gather_heatmaps.py, with:
  - Section/row-based prompt naming (from prompts.py).
  - Token axes decorated with structural ID labels (e.g. "John_1", "consoles_2")
    using the hi_frame_tagged stored in each cached bundle.  The IDs come from
    the grammar's constituent numbering, so both the HI and HF axes show the same
    ID for the same word regardless of its linear position.

HOW TO RUN
----------
After Stage 1 has populated all caches:
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/gather_heatmaps.py
or as a single CPU job:
    sbatch slurm/run_cpu.sbatch \\
        experiments/15_multi_seed_grammar_v2_depth_2_random/gather_heatmaps.py
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from compare_heatmaps import (
    align_heads_to_reference,
    plot_grid,
    _file_suffix,
    parse_seeds,
)
from prompts import PROMPTS, prompt_name, build_word_id_map, label_tokens

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
ARCHS       = ["vaswani", "vaswani_rope", "gpt2_rope"]


def cache_path(arch: str, seed: int, section: str, row: int) -> Path:
    return RESULTS_DIR / arch / f"seed_{seed}" / "attn" / f"{prompt_name(section, row)}.pt"


def load_bundles(arch: str, seeds: list[int], section: str, row: int) -> list[dict]:
    bundles = []
    for seed in seeds:
        path = cache_path(arch, seed, section, row)
        if not path.exists():
            sys.exit(
                f"missing cache: {path}\n"
                f"  run Stage 1 first:\n"
                f"  sbatch --array=0-{len(seeds) * len(ARCHS) - 1}%9 "
                f"slurm/run_gpu.sbatch "
                f"experiments/15_multi_seed_grammar_v2_depth_2_random/extract_attention.py"
            )
        bundles.append(torch.load(path, weights_only=False))
    return bundles


def render_prompt(arch, seeds, ref_idx, align, section, row) -> list[tuple]:
    bundles = load_bundles(arch, seeds, section, row)
    n_seeds = len(seeds)
    families = list(bundles[0]["groups"].keys())
    n_layers = bundles[0]["groups"][families[0]].shape[0]

    for family in families:
        ref_shape = bundles[0]["groups"][family].shape
        for seed, b in zip(seeds, bundles):
            if b["groups"][family].shape != ref_shape:
                sys.exit(
                    f"shape mismatch {arch} {prompt_name(section, row)} seed {seed} "
                    f"family '{family}': {tuple(b['groups'][family].shape)} != "
                    f"{tuple(ref_shape)}"
                )

    word_to_id = build_word_id_map(bundles[0]["hi"], bundles[0]["hi_frame_tagged"])

    out_root = FIGURES_DIR / arch / "heatmaps" / prompt_name(section, row)
    out_root.mkdir(parents=True, exist_ok=True)
    nonref = [si for si in range(n_seeds) if si != ref_idx]
    rows: list[tuple] = []

    for family in families:
        per_seed = [b["groups"][family] for b in bundles]
        n_heads = per_seed[0].shape[1]
        qtok = label_tokens(bundles[0]["query_tokens"][family], word_to_id)
        ktok = label_tokens(bundles[0]["key_tokens"][family], word_to_id)

        for layer in range(n_layers):
            vecs = [per_seed[s][layer].reshape(n_heads, -1) for s in range(n_seeds)]
            perms, matched, naive = align_heads_to_reference(vecs, ref_idx)

            oi = sum(sum(matched[s]) / n_heads for s in nonref) / max(len(nonref), 1)
            nv = sum(naive[s] for s in nonref) / max(len(nonref), 1)
            for s in range(n_seeds):
                rows.append((section, row, family, layer, seeds[s], seeds[ref_idx],
                             sum(matched[s]) / n_heads, naive[s]))

            save_path = out_root / f"layer_{layer}_{_file_suffix(family)}.png"
            title = (
                f"{arch} | {prompt_name(section, row)} | {family} | layer {layer} | "
                f"rows=seeds cols=heads | order-inv cos={oi:.3f} (naive {nv:.3f})"
            )
            plot_grid(per_seed, seeds, layer, qtok, ktok, title, save_path,
                      perms, matched, ref_idx, align)
            print(
                f"  wrote {save_path.relative_to(EXP_DIR)}  "
                f"(order-inv={oi:.3f}, naive={nv:.3f})",
                flush=True,
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=ARCHS, default=None,
                   help="Restrict to one arch. Default: all three.")
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--align-heads", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reference-seed", type=int, default=None)
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    ref_idx = 0
    if args.reference_seed is not None:
        if args.reference_seed not in seeds:
            sys.exit(f"--reference-seed {args.reference_seed} not in --seeds {seeds}.")
        ref_idx = seeds.index(args.reference_seed)

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        print(
            f"\n=== {arch}  seeds={seeds}  reference=seed {seeds[ref_idx]}  "
            f"align={args.align_heads} ===",
            flush=True,
        )
        arch_rows: list[tuple] = []
        for prompt in PROMPTS:
            name = prompt_name(prompt["section"], prompt["row"])
            print(f"[{name}] {prompt['hi']!r}", flush=True)
            arch_rows.extend(
                render_prompt(arch, seeds, ref_idx, args.align_heads,
                              prompt["section"], prompt["row"])
            )

        csv_path = RESULTS_DIR / arch / "heatmap_consistency.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "row", "family", "layer", "seed", "ref_seed",
                        "mean_matched_cosine", "mean_naive_cosine"])
            for section, row, family, layer, seed, ref_seed, mm, mn in arch_rows:
                w.writerow([section, row, family, layer, seed, ref_seed,
                            f"{mm:.4f}", f"{mn:.4f}"])
        print(
            f"  metrics -> {csv_path.relative_to(EXP_DIR)} ({len(arch_rows)} rows)",
            flush=True,
        )


if __name__ == "__main__":
    main()
