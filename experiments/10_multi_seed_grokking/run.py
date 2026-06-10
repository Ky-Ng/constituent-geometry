"""Seed-sweep orchestrator for experiment 10 (multi-seed GROKKING on held-out depth).

WHY
---
Experiment 09 re-ran the 06/07/08 trainings across seeds for only 30 epochs on
the `random_frame` depth-3 dataset -- where the models already hit ~1.0 test
exact-match, so there is no generalization gap to "grok". 10 moves to the
`held_out_depth` dataset (train sees depths 0-2; val/test are unseen depth-3
frames) and trains for *many* more epochs (default 3000) to look for grokking:
train loss collapses early (memorization) while held-out exact-match stays low
and then, possibly, jumps much later (delayed generalization).

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
Like 09, it is a thin launcher that shells out to per-arch training scripts, one
subprocess per seed. The ONLY differences from 09:
  - It calls *local* runner copies under `runners/` (verbatim copies of
    06/07/08 with a single added `--weight-decay` knob -- the main grokking
    lever, which 06/07/08 hardcode to 0.0). We copy rather than edit the
    originals per repo convention.
  - It threads four training knobs through to each child: `--weight-decay`,
    `--lr-scheduler`, `--eval-steps`, `--save-steps`.
      * `--lr-scheduler` defaults to 'constant_with_warmup' (flat LR after
        warmup), NOT 06/07/08's 'linear' decay-to-0: a decaying LR is ~0 by the
        late epochs where grokking would occur, and a constant LR also lets the
        run be extended cleanly later.
      * `--save-steps` is set LARGE on purpose: at 3000 epochs (~198k steps on
        this dataset) the child's default save_steps=200 with no
        save_total_limit would write ~990 checkpoint dirs per run. We can't add
        save_total_limit without editing the children, so we cut checkpoint
        count by saving rarely (default ~20/run) while still evaluating often.
  - Output dir, run name, wandb GROUP, and tags all encode (epochs, weight
    decay) so sweeping either knob lands in its own bucket instead of merging.

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (DEFAULT for 10) -- one job per architecture packs all seeds onto the
  single allocated GPU, throttled by --max-parallel. Models are tiny (~1.8M
  params), so ~5 fit comfortably on one 48 GB A6000:
      sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch gpt2_rope

* Array (1 GPU per run) -- if SLURM_ARRAY_TASK_ID is set, run exactly ONE
  (arch, seed) pair. With N seeds and 3 archs, use --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-14 slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py

LOCAL SMOKE TEST (2 seeds, 1 epoch, frequent eval/save, no wandb / no Hub push)
-------------------------------------------------------------------------------
    uv run python experiments/10_multi_seed_grokking/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 \\
        --eval-steps 20 --save-steps 20 --weight-decay 0.1 \\
        --max-parallel 2 --no-wandb
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]            # experiments/10_multi_seed_grokking -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the LOCAL runner copy we drive (06/07/08 + a --weight-decay knob).
# These live under this experiment so 10 is self-contained and reproducible.
ARCH_RUNNERS: dict[str, Path] = {
    "vaswani":      EXP_DIR / "runners/vaswani_run.py",
    "vaswani_rope": EXP_DIR / "runners/vaswani_rope_run.py",
    "gpt2_rope":    EXP_DIR / "runners/gpt2_rope_run.py",
}
ARCHS = list(ARCH_RUNNERS)                # stable order for array-mode indexing

# wandb grouping: every seed of one (arch, epochs, wd) setting shares a run
# GROUP, so the wandb UI collapses them into one panel with a mean curve +
# min/max band across seeds. We fold epochs AND weight decay into the group/tag
# names so a later longer-epoch or different-wd sweep does NOT merge into the
# same band. Passed via env vars that wandb.init() / the HF Trainer honor
# automatically -- no edits to the runners needed.
WANDB_PREFIX = "10_multi_seed_grokking"


def parse_seeds(spec: str) -> list[int]:
    """'42-46' -> [42..46] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def tag(epochs: int, wd: float) -> str:
    """Shared (epochs, wd) suffix used in dir / run-name / group / tags."""
    return f"ep{epochs}_wd{wd}"


def build_cmd(arch: str, seed: int, epochs: int, dataset: str, wd: float,
              lr_scheduler: str, eval_steps: int, save_steps: int, no_wandb: bool):
    """Construct the argv for one (arch, seed) training and its output dir / run name."""
    suffix = tag(epochs, wd)
    out_dir = RESULTS_DIR / arch / suffix / f"seed_{seed}"
    run_name = f"{WANDB_PREFIX}_{arch}_{suffix}_seed_{seed}"
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--weight-decay", str(wd),
        "--lr-scheduler", lr_scheduler,
        "--eval-steps", str(eval_steps),
        "--save-steps", str(save_steps),
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",                       # keep checkpoints local; do not create Hub repos
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd, out_dir, run_name


def _launch(arch: str, seed: int, epochs: int, dataset: str, wd: float,
            lr_scheduler: str, eval_steps: int, save_steps: int, no_wandb: bool):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(
        arch, seed, epochs, dataset, wd, lr_scheduler, eval_steps, save_steps, no_wandb
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Group this (arch, epochs, wd)'s seeds together in wandb; tag for filtering.
    # Harmless when --no-wandb (the child sets report_to="none" and never inits).
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{arch}_{tag(epochs, wd)}"
    env["WANDB_TAGS"] = f"{WANDB_PREFIX},{arch},seed_{seed},ep{epochs},wd{wd}"
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, epochs, dataset, wd, lr_scheduler, eval_steps, save_steps,
               no_wandb, max_parallel) -> list[str]:
    """Run all (arch, seed) jobs as concurrent subprocesses, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            a, s = pending.pop(0)
            running.append(_launch(a, s, epochs, dataset, wd,
                                    lr_scheduler, eval_steps, save_steps, no_wandb))
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


def run_one(arch, seed, epochs, dataset, wd, lr_scheduler, eval_steps, save_steps,
            no_wandb) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(
        arch, seed, epochs, dataset, wd, lr_scheduler, eval_steps, save_steps, no_wandb
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
    p.add_argument("--seeds", default="42-46",
                   help="Inclusive range 'A-B' or comma list. Default 42-46 (5 seeds).")
    p.add_argument("--epochs", type=int, default=3000,
                   help="Long horizon to look for grokking (delayed generalization).")
    p.add_argument("--dataset", default="kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3",
                   help="held_out_depth depth-3 dataset: train depths 0-2, val/test unseen depth 3.")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW L2 -- the main grokking lever (06/07/08 hardcode 0.0). "
                        "Try 0.01 / 0.1 / 1.0; each value lands in its own wandb group + results dir.")
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   help="HF lr_scheduler_type. Default 'constant_with_warmup' (flat LR after "
                        "warmup), NOT 06/07/08's 'linear' decay-to-0: a decaying LR is ~0 by the "
                        "late epochs where grokking would occur, and constant LR also lets runs "
                        "extend cleanly. Pass 'linear'/'cosine' to override.")
    p.add_argument("--eval-steps", type=int, default=330,
                   help="~every 5 epochs on this dataset (66 steps/epoch) -> ~600 eval points "
                        "across 3000 epochs: enough resolution to see a grok transition.")
    p.add_argument("--save-steps", type=int, default=9900,
                   help="Kept LARGE (30x eval_steps) so 3000 epochs writes ~20 checkpoints/run "
                        "instead of ~990. Must stay a multiple of --eval-steps "
                        "(HF requires this under load_best_model_at_end=True).")
    p.add_argument("--max-parallel", type=int, default=5,
                   help="Packed-mode concurrency on the single allocated GPU (5 seeds).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Pass through to each runner (skip wandb logging).")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)

    if args.save_steps % args.eval_steps != 0:
        sys.exit(f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
                 f"({args.eval_steps}); HF requires this when load_best_model_at_end=True.")

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
        sys.exit(run_one(arch, seed, args.epochs, args.dataset, args.weight_decay,
                         args.lr_scheduler, args.eval_steps, args.save_steps, args.no_wandb))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"epochs={args.epochs} wd={args.weight_decay} lr_scheduler={args.lr_scheduler} "
          f"eval_steps={args.eval_steps} save_steps={args.save_steps} "
          f"max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset, args.weight_decay,
                          args.lr_scheduler, args.eval_steps, args.save_steps,
                          args.no_wandb, args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
