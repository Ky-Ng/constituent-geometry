"""Stage 1 of the experiment-15 cross-seed comparison: EXTRACT & CACHE attention.

Mirrors experiments/09_multi_seed/extract_attention.py, with three differences:
  - Loads each run's best/ model (the early-stopped checkpoint selected on val
    exact-match). With per-seed early stopping there is no common training step,
    so best/ is the natural comparison point (all runs converge to ~0.99).
  - Defaults to --device cuda (GPU array job).
  - Cache bundles also store section, row, hi_frame_tagged, hf_frame_tagged
    so gather_heatmaps.py can attach structural ID labels to token axes.

CACHE LAYOUT
------------
    results/<arch>/seed_<S>/attn/<section>-<row>.pt

HOW TO RUN
----------
GPU array (30 tasks, at most 9 concurrent):
    sbatch --array=0-29%9 slurm/run_gpu.sbatch \\
        experiments/15_multi_seed_grammar_v2_depth_2_random/extract_attention.py

Single smoke test (one arch / seed):
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/extract_attention.py \\
        --arch vaswani --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from visualization.attention_heatmaps import ADAPTERS
from prompts import PROMPTS, prompt_name

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
ARCHS       = ["vaswani", "vaswani_rope", "gpt2_rope"]
CHECKPOINT  = "best"


def parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def cache_path(arch: str, seed: int, section: str, row: int) -> Path:
    return RESULTS_DIR / arch / f"seed_{seed}" / "attn" / f"{prompt_name(section, row)}.pt"


def extract_seed(arch: str, seed: int, device: str, overwrite: bool) -> None:
    ckpt = RESULTS_DIR / arch / f"seed_{seed}" / CHECKPOINT
    if not ckpt.exists():
        sys.exit(f"missing checkpoint: {ckpt}")

    load, extract = ADAPTERS[arch]["load"], ADAPTERS[arch]["extract"]
    model, tok = load(str(ckpt), device)
    print(f"[load] {arch} seed {seed}  (checkpoint={CHECKPOINT})", flush=True)

    for p in PROMPTS:
        out = cache_path(arch, seed, p["section"], p["row"])
        if out.exists() and not overwrite:
            print(f"  skip (cached) {out.relative_to(EXP_DIR)}", flush=True)
            continue
        bundle = extract(model, tok, p["hi"], p["hf"], device, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "groups":           bundle.groups,
                "query_tokens":     bundle.query_tokens,
                "key_tokens":       bundle.key_tokens,
                "hi":               p["hi"],
                "hf":               p["hf"],
                "section":          p["section"],
                "row":              p["row"],
                "hi_frame_tagged":  p["hi_frame_tagged"],
                "hf_frame_tagged":  p["hf_frame_tagged"],
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
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--arch", choices=ARCHS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda", help="cpu | cuda (default: cuda)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)

    task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_env is not None:
        task = int(task_env)
        n = len(seeds)
        if task >= len(ARCHS) * n:
            sys.exit(
                f"SLURM_ARRAY_TASK_ID={task} out of range "
                f"(use --array=0-{len(ARCHS) * n - 1})."
            )
        arch, seed = ARCHS[task // n], seeds[task % n]
        print(f"[array] task {task} -> arch={arch} seed={seed}", flush=True)
        extract_seed(arch, seed, args.device, args.overwrite)
        return

    if args.seed is not None:
        if args.arch is None:
            sys.exit("--seed requires --arch.")
        extract_seed(args.arch, args.seed, args.device, args.overwrite)
        return

    archs = [args.arch] if args.arch else ARCHS
    for arch in archs:
        for seed in seeds:
            extract_seed(arch, seed, args.device, args.overwrite)
    print("\nAll extractions complete.", flush=True)


if __name__ == "__main__":
    main()
