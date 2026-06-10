"""Stage 1 of the experiment-09 cross-seed comparison: EXTRACT & CACHE attention.

WHY TWO STAGES
--------------
A cross-seed grid needs every seed's attention for ONE prompt side by side, so a
single figure cannot be split across seeds. But the EXPENSIVE part -- loading each
trained model and running a forward pass -- is per (arch, seed) and embarrassingly
parallel. So we split the work:

  Stage 1 (THIS file): one job per (arch, seed). Load that model ONCE, forward all
    12 PROMPTS, and cache each prompt's AttentionBundle to disk. 3 archs x 10 seeds
    = 30 independent CPU jobs.
  Stage 2 (gather_heatmaps.py): read the cached bundles (NO model loads) and build
    the S x H cosine-reordered grids + the consistency CSV.

Caching is also the foundation for the planned similarity-metric iteration: any
future metric is just another cheap pass over the cached `.pt` files -- no
re-inference, no GPU.

The models are tiny (~1.8M params), so CPU inference is fast; we ask for NO GPU.
Loading + extraction are reused verbatim from src/visualization/attention_heatmaps.py
(ADAPTERS) -- no model logic is duplicated here.

CACHE LAYOUT
------------
    results/<arch>/seed_<S>/attn/depth<D>-ex<N>.pt
Each file is a dict: {groups, query_tokens, key_tokens, hi, hf, depth, ex}, where
`groups` maps family -> attention tensor [L, H, Tq, Tk] (already on CPU).

HOW TO RUN
----------
* Cluster (the per-seed parallelism), 30-task CPU array:
      sbatch --array=0-29 slurm/run_cpu.sbatch \\
          experiments/09_multi_seed/extract_attention.py
  task -> (arch, seed):  arch = ARCHS[task // n_seeds];  seed = seeds[task % n_seeds]

* Local, everything in one process (slow but simple):
      uv run python experiments/09_multi_seed/extract_attention.py

* Local smoke test, one (arch, seed):
      uv run python experiments/09_multi_seed/extract_attention.py --arch vaswani --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Same arch keys as ADAPTERS; loader/extractor come straight from the shared module.
from visualization.attention_heatmaps import ADAPTERS

# This file's directory is on sys.path when run as a script, so `prompts` imports.
from prompts import PROMPTS, prompt_name

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]   # stable order for array indexing


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def cache_path(arch: str, seed: int, depth: int, ex: int) -> Path:
    return RESULTS_DIR / arch / f"seed_{seed}" / "attn" / f"{prompt_name(depth, ex)}.pt"


def extract_seed(arch: str, seed: int, device: str, overwrite: bool) -> None:
    """Load one (arch, seed) checkpoint once; cache all PROMPTS' attention bundles."""
    ckpt = RESULTS_DIR / arch / f"seed_{seed}" / "best"
    if not ckpt.exists():
        sys.exit(f"missing checkpoint: {ckpt} (run the seed sweep run.py first)")

    load, extract = ADAPTERS[arch]["load"], ADAPTERS[arch]["extract"]
    model, tok = load(str(ckpt), device)
    print(f"[load] {arch} seed {seed}  ({ckpt.relative_to(EXP_DIR)})", flush=True)

    for p in PROMPTS:
        out = cache_path(arch, seed, p["depth"], p["ex"])
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
    p.add_argument("--arch", choices=ARCHS, default=None,
                   help="Restrict to one arch (local use). Default: all three.")
    p.add_argument("--seed", type=int, default=None,
                   help="Restrict to one seed (local smoke test). Requires --arch.")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps (cpu is plenty).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if a cache file already exists.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)

    # --- Array mode: SLURM_ARRAY_TASK_ID selects exactly one (arch, seed) -----
    task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_env is not None:
        task = int(task_env)
        n = len(seeds)
        if task >= len(ARCHS) * n:
            sys.exit(f"SLURM_ARRAY_TASK_ID={task} out of range for "
                     f"{len(ARCHS)} archs x {n} seeds (use --array=0-{len(ARCHS) * n - 1}).")
        arch, seed = ARCHS[task // n], seeds[task % n]
        print(f"[array] task {task} -> arch={arch} seed={seed}", flush=True)
        extract_seed(arch, seed, args.device, args.overwrite)
        return

    # --- Local mode: one (arch, seed), or sweep everything --------------------
    if args.seed is not None:
        if args.arch is None:
            sys.exit("--seed requires --arch (pick one cell to extract).")
        extract_seed(args.arch, args.seed, args.device, args.overwrite)
        return

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        for seed in seeds:
            extract_seed(arch, seed, args.device, args.overwrite)
    print("\nAll extractions complete.", flush=True)


if __name__ == "__main__":
    main()
