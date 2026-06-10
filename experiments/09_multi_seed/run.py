"""Seed-sweep orchestrator for experiment 09 (multi-seed attention-heatmap consistency).

WHY
---
Experiments 06/07/08 each train a *single* model (seed 0) on the `random_frame`
depth-3 HI->HF dataset and visualize its attention heatmaps. We want to know
whether those heatmaps are an artifact of the seed or a robust property of the
architecture/task. So 09 re-runs the SAME three trainings across many seeds
(default 42-51) and saves all checkpoints locally for a later cross-seed
comparison (deferred to a follow-up script).

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
It does NOT redefine any model or training logic. It is a thin launcher that
shells out to the existing experiment scripts, one subprocess per seed:
  - experiments/06_.../run.py   (arch key: vaswani,      sinusoidal PE)
  - experiments/07_.../run.py   (arch key: vaswani_rope, RoPE seq2seq)
  - experiments/08_.../run.py   (arch key: gpt2_rope,    decoder-only RoPE)
Each is invoked with a different --seed, --no-push (so we do NOT create 30 HF
Hub repos), and an --output-dir under experiments/09_multi_seed/results/<arch>/seed_<S>/.
Everything else (tokenizer, max-length, optimizer, eval wiring) stays at each
script's own defaults, so 09 reproduces 06/07/08 exactly except for the seed.

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (default) -- one job per architecture packs all seeds onto the single
  allocated GPU, throttled by --max-parallel. The models are tiny (~1.8M params,
  well under 1 GB each), so ~10 fit on one 48 GB A6000 and interleave well (each
  run is latency/CPU-bound, so the GPU is mostly idle for any single run):
      sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch gpt2_rope

* Array (1 GPU per run) -- if SLURM_ARRAY_TASK_ID is set, run exactly ONE
  (arch, seed) pair. With N seeds and 3 archs, use --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-29 slurm/run_gpu.sbatch experiments/09_multi_seed/run.py

LOCAL SMOKE TEST (2 seeds, 1 epoch, no wandb / no Hub push)
-----------------------------------------------------------
    uv run python experiments/09_multi_seed/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]            # experiments/09_multi_seed -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the existing experiment run.py we reuse (one source of truth).
ARCH_RUNNERS: dict[str, Path] = {
    "vaswani":      REPO_ROOT / "experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py",
    "vaswani_rope": REPO_ROOT / "experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py",
    "gpt2_rope":    REPO_ROOT / "experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py",
}
ARCHS = list(ARCH_RUNNERS)                # stable order for array-mode indexing

# wandb grouping: every seed of one arch shares a run GROUP, so the wandb UI
# collapses them into one panel with a mean curve + min/max band across seeds.
# A shared TAG lets you pull up all 30 runs at once. Both are passed via env
# vars that wandb.init() (and the HF Trainer's wandb integration) honor
# automatically -- no edits to 06/07/08 needed.
WANDB_PREFIX = "09_multi_seed"


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def build_cmd(arch: str, seed: int, epochs: int, dataset: str, no_wandb: bool):
    """Construct the argv for one (arch, seed) training and its output dir / run name."""
    out_dir = RESULTS_DIR / arch / f"seed_{seed}"
    run_name = f"09_multi_seed_{arch}_seed_{seed}"
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",                       # keep checkpoints local; do not create Hub repos
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd, out_dir, run_name


def _launch(arch: str, seed: int, epochs: int, dataset: str, no_wandb: bool):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(arch, seed, epochs, dataset, no_wandb)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Group this arch's seeds together in wandb; tag all 30 with a shared label.
    # Harmless when --no-wandb (the child sets report_to="none" and never inits).
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{arch}"
    env["WANDB_TAGS"] = f"{WANDB_PREFIX},{arch},seed_{seed}"
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, epochs, dataset, no_wandb, max_parallel) -> list[str]:
    """Run all (arch, seed) jobs as concurrent subprocesses, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            running.append(_launch(*pending.pop(0), epochs, dataset, no_wandb))
        time.sleep(2)
        still = []
        for proc, run_name, log in running:
            if proc.poll() is None:
                still.append((proc, run_name, log))
                continue
            log.close()
            ok = proc.returncode == 0
            print(f"[done ] {run_name}: {'ok' if ok else f'FAILED (exit {proc.returncode})'}",
                  flush=True)
            if not ok:
                failures.append(run_name)
        running = still
    return failures


def run_one(arch, seed, epochs, dataset, no_wandb) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(arch, seed, epochs, dataset, no_wandb)
    rc = proc.wait()
    log.close()
    print(f"[done ] {run_name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=[*ARCHS, "all"], default="all",
                   help="Which architecture to sweep in packed mode (ignored in array mode, "
                        "which always sweeps all three).")
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51 (10 seeds).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Match the 06/07/08 baseline (30) so heatmaps stay comparable.")
    p.add_argument("--dataset", default="kylelovesllms/hi_hf_frames_d3_random_100")
    p.add_argument("--max-parallel", type=int, default=10,
                   help="Packed-mode concurrency on the single allocated GPU.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Pass through to each run.py (skip wandb logging).")
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
        sys.exit(run_one(arch, seed, args.epochs, args.dataset, args.no_wandb))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset, args.no_wandb, args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
