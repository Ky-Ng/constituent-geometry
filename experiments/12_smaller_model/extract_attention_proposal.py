"""Stage 1 for experiment 12: EXTRACT & CACHE attention (the 09 pipeline, re-pathed).

WHAT THIS IS
------------
A near-verbatim copy of `experiments/09_multi_seed/extract_attention.py`, with the
ONLY change being the path layout: experiment 12 stores checkpoints under an extra
`L<l>_H<h>` SHAPE level

    results/<arch>/<shape>/seed_<S>/best          (12)   vs
    results/<arch>/seed_<S>/best                  (09)

so every path here threads a `shape` between `<arch>` and `seed_<S>`, and the
launcher sweeps BOTH shapes (`L2_H2`, `L3_H2`) by default. All MODEL LOGIC is
imported, not duplicated:
  - loaders/extractors come from `visualization.attention_heatmaps.ADAPTERS`
    (`from_pretrained` reads n_layers/n_heads from each checkpoint's config, so the
    SAME loader handles both shapes -- nothing shape-specific here).
  - the 12 depth-stratified prompts come from experiment 09's `prompts.py`
    (same dataset, same gold pairs -> directly comparable to 09's figures).
We add 09's directory to sys.path so `from prompts import ...` resolves to it.

CACHE LAYOUT
------------
    results/<arch>/<shape>/seed_<S>/attn/depth<D>-ex<N>.pt
Each file: {groups, query_tokens, key_tokens, hi, hf, depth, ex}, where `groups`
maps family -> attention tensor [L, H, Tq, Tk] (CPU). Stage 2 (gather) reads these.

HOW TO RUN
----------
* Cluster, 60-task CPU array (3 archs x 2 shapes x 10 seeds):
      sbatch --array=0-59 slurm/run_cpu.sbatch \\
          experiments/12_smaller_model/extract_attention.py
  task -> (arch, shape, seed):
      arch  = ARCHS[task // (n_shapes * n_seeds)]
      rem   = task %  (n_shapes * n_seeds)
      shape = SHAPES[rem // n_seeds]
      seed  = seeds[rem %  n_seeds]

* Local, everything in one process (tiny models -> CPU is plenty):
      uv run python experiments/12_smaller_model/extract_attention.py

* Local smoke test, one cell:
      uv run python experiments/12_smaller_model/extract_attention.py \\
          --arch vaswani --shapes L2_H2 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
NINE_DIR = EXP_DIR.parent / "09_multi_seed"      # reuse 09's prompts (one source of truth)
sys.path.insert(0, str(NINE_DIR))

from visualization.attention_heatmaps import ADAPTERS   # noqa: E402
from prompts import PROMPTS, prompt_name                # noqa: E402  (from 09 dir)

RESULTS_DIR = EXP_DIR / "results"
ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]        # stable order for array indexing
SHAPES = ["L2_H2", "L3_H2"]                             # stable order for array indexing


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def parse_shapes(spec: str) -> list[str]:
    """'L2_H2,L3_H2' -> ['L2_H2', 'L3_H2'] (comma list, order preserved)."""
    return [s.strip() for s in spec.split(",") if s.strip()]


def cache_path(arch: str, shape: str, seed: int, depth: int, ex: int) -> Path:
    return RESULTS_DIR / arch / shape / f"seed_{seed}" / "attn" / f"{prompt_name(depth, ex)}.pt"


def extract_seed(arch: str, shape: str, seed: int, device: str, overwrite: bool) -> None:
    """Load one (arch, shape, seed) checkpoint once; cache all PROMPTS' bundles."""
    ckpt = RESULTS_DIR / arch / shape / f"seed_{seed}" / "best"
    if not ckpt.exists():
        sys.exit(f"missing checkpoint: {ckpt} (run the seed sweep run.py first)")

    load, extract = ADAPTERS[arch]["load"], ADAPTERS[arch]["extract"]
    model, tok = load(str(ckpt), device)
    print(f"[load] {arch} {shape} seed {seed}  ({ckpt.relative_to(EXP_DIR)})", flush=True)

    for p in PROMPTS:
        out = cache_path(arch, shape, seed, p["depth"], p["ex"])
        if out.exists() and not overwrite:
            print(f"  skip (cached) {out.relative_to(EXP_DIR)}", flush=True)
            continue
        # hf is the GOLD target string, so max_new_tokens is unused (no generation).
        bundle = extract(model, tok, p["hi"], p["hf"], device, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "groups": bundle.groups,
                "query_tokens": bundle.query_tokens,
                "key_tokens": bundle.key_tokens,
                "hi": p["hi"],
                "hf": p["hf"],
                "depth": p["depth"],
                "ex": p["ex"],
            },
            out,
        )
        print(f"  wrote {out.relative_to(EXP_DIR)}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51 (10 seeds).")
    p.add_argument("--shapes", default="L2_H2,L3_H2",
                   help="Comma list of shape dirs to sweep. Default both 12 shapes.")
    p.add_argument("--arch", choices=ARCHS, default=None,
                   help="Restrict to one arch (local use). Default: all three.")
    p.add_argument("--seed", type=int, default=None,
                   help="Restrict to one seed (local smoke test). Requires --arch.")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps (cpu is plenty).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if a cache file already exists.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    shapes = parse_shapes(args.shapes)

    # --- Array mode: SLURM_ARRAY_TASK_ID selects one (arch, shape, seed) ------
    task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_env is not None:
        task = int(task_env)
        n, m = len(seeds), len(shapes)
        total = len(ARCHS) * m * n
        if task >= total:
            sys.exit(f"SLURM_ARRAY_TASK_ID={task} out of range for "
                     f"{len(ARCHS)} archs x {m} shapes x {n} seeds "
                     f"(use --array=0-{total - 1}).")
        arch = ARCHS[task // (m * n)]
        rem = task % (m * n)
        shape, seed = shapes[rem // n], seeds[rem % n]
        print(f"[array] task {task} -> arch={arch} shape={shape} seed={seed}", flush=True)
        extract_seed(arch, shape, seed, args.device, args.overwrite)
        return

    # --- Local mode: one cell, or sweep everything ----------------------------
    if args.seed is not None:
        if args.arch is None:
            sys.exit("--seed requires --arch (pick one cell to extract).")
        for shape in shapes:
            extract_seed(args.arch, shape, args.seed, args.device, args.overwrite)
        return

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        for shape in shapes:
            for seed in seeds:
                extract_seed(arch, shape, seed, args.device, args.overwrite)
    print("\nAll extractions complete.", flush=True)


if __name__ == "__main__":
    main()
