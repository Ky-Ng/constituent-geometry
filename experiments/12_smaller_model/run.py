"""Seed-sweep orchestrator for experiment 12 (SMALLER model: 3 layers, 2 heads).

WHY
---
Experiment 09 swept seeds 42-51 of the 06/07/08 trainings at the baseline shape
`n_layers=4, n_heads=4` and asked whether the attention heatmaps are seed-robust.
12 asks the *capacity* question on the same task: does a deliberately SMALLER
model -- `n_layers=3, n_heads=2` -- still (a) solve the `random_frame` depth-3
HI->HF task to ~1.0 test exact-match, and (b) produce the same cross-seed heatmap
consistency? Fewer heads also makes the head-matching analysis cheaper/cleaner
(2! = 2 pairings per layer instead of 4! = 24).

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
Exactly like 09: a thin launcher that shells out to the EXISTING 06/07/08 scripts,
one subprocess per seed. It does NOT redefine any model or training logic.

The ONLY difference from 09 is two extra flags threaded to each child:
  --n-layers (default 3) and --n-heads (default 2).
06/07/08 already expose both as argparse knobs (defaults 4/4), so 12 needs NO
local runner copies -- unlike 10, which had to copy runners/ to add a brand-new
--weight-decay knob. We just pass the smaller shape through:
  - experiments/06_.../run.py   (arch key: vaswani,      sinusoidal PE)
  - experiments/07_.../run.py   (arch key: vaswani_rope, RoPE seq2seq)
  - experiments/08_.../run.py   (arch key: gpt2_rope,    decoder-only RoPE)
Everything else (dataset, tokenizer, max-length, optimizer, eval wiring, 30
epochs) stays at 09's values, so 12 is 09 with a smaller model -- nothing more.

Output dir, run name, wandb GROUP, and tags all encode the shape (L<l>_H<h>) so a
12-run never merges with a 09-run (or with a different shape if you sweep L/H).

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (default) -- one job per architecture packs all seeds onto the single
  allocated GPU, throttled by --max-parallel. The models are even tinier than
  09's (~1.8M -> ~1.3M params), so ~10 fit on one 48 GB A6000 and interleave well
  (each run is latency/CPU-bound, so the GPU is mostly idle for any single run):
      sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch gpt2_rope

* Array (1 GPU per run) -- if SLURM_ARRAY_TASK_ID is set, run exactly ONE
  (arch, seed) pair. With N seeds and 3 archs, use --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-29 slurm/run_gpu.sbatch experiments/12_smaller_model/run.py

LOCAL SMOKE TEST (2 seeds, 1 epoch, no wandb / no Hub push)
-----------------------------------------------------------
    uv run python experiments/12_smaller_model/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]            # experiments/12_smaller_model -> experiments -> repo root
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
# A shared TAG lets you pull up all runs at once. Both are passed via env vars
# that wandb.init() (and the HF Trainer's wandb integration) honor automatically
# -- no edits to 06/07/08 needed.
WANDB_PREFIX = "12_smaller_model"


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def build_cmd(arch, seed, epochs, dataset, n_layers, n_heads, no_wandb):
    """Construct the argv for one (arch, seed) training and its output dir / run name.

    The run name and output dir encode the shape (L<l>_H<h>) so 12's smaller-model
    runs never collide with 09's L4_H4 runs in results/ or wandb.
    """
    shape = f"L{n_layers}_H{n_heads}"
    out_dir = RESULTS_DIR / arch / shape / f"seed_{seed}"
    run_name = f"12_smaller_model_{arch}_{shape}_seed_{seed}"
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--n-layers", str(n_layers),       # 06/07/08 default is 4; 12 shrinks to 3
        "--n-heads", str(n_heads),         # 06/07/08 default is 4; 12 shrinks to 2
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",                       # keep checkpoints local; do not create Hub repos
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd, out_dir, run_name


def _launch(arch, seed, epochs, dataset, n_layers, n_heads, no_wandb):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(
        arch, seed, epochs, dataset, n_layers, n_heads, no_wandb
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Group this arch's seeds together in wandb; tag all runs with a shared label
    # that also encodes the shape. Harmless when --no-wandb (the child sets
    # report_to="none" and never inits).
    shape = f"L{n_layers}_H{n_heads}"
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{arch}_{shape}"
    env["WANDB_TAGS"] = f"{WANDB_PREFIX},{arch},{shape},seed_{seed}"
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, epochs, dataset, n_layers, n_heads, no_wandb, max_parallel) -> list[str]:
    """Run all (arch, seed) jobs as concurrent subprocesses, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            arch, seed = pending.pop(0)
            running.append(
                _launch(arch, seed, epochs, dataset, n_layers, n_heads, no_wandb)
            )
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


def run_one(arch, seed, epochs, dataset, n_layers, n_heads, no_wandb) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(
        arch, seed, epochs, dataset, n_layers, n_heads, no_wandb
    )
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
                   help="Match the 09 baseline (30) so heatmaps stay comparable.")
    p.add_argument("--dataset", default="kylelovesllms/hi_hf_frames_d3_random_100")
    # --- the smaller-model knobs (the whole point of 12) --------------------
    p.add_argument("--n-layers", type=int, default=3,
                   help="Encoder/decoder layers per child (09 baseline = 4).")
    p.add_argument("--n-heads", type=int, default=2,
                   help="Attention heads per layer per child (09 baseline = 4).")
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
        print(f"[array] task {task} -> arch={arch} seed={seed} "
              f"L={args.n_layers} H={args.n_heads}", flush=True)
        sys.exit(run_one(arch, seed, args.epochs, args.dataset,
                         args.n_layers, args.n_heads, args.no_wandb))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"L={args.n_layers} H={args.n_heads} max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset,
                          args.n_layers, args.n_heads, args.no_wandb, args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
