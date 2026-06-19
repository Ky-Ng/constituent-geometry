"""Seed-sweep GROKKING-probe config on the v2 grammar's *random* depth-2 split (experiment 18).

WHY
---
This is the **in-distribution control** for experiment 16. Exp 16 ran the
grokking probe (no early-stop, constant LR, wd=1.0, 5 seeds) on the v2 grammar's
**held-out-DEPTH-2** split -- train sees only shallow trees, val/test are DEEPER
held-out structures -- to ask whether depth extrapolation is recovered by long
training + weight decay. 18 runs the *identical probe machinery* on the v2
grammar's **random-frame** depth-2 split instead: train/val/test are
frame-disjoint but drawn from the SAME depth-{0,1,2} distribution, so there is no
structural gap to extrapolate across. It measures how the same config behaves
when generalization is in-distribution -- the baseline against which 16's
extrapolation result is read.

  Dataset: kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random
  (full depth-2 frame enumeration ~1.39M frames x k=2 lexicalizations ->
   train 2,229,638 / val 278,704 / test 278,706 rows; random 80/10/10 over frames.)

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
A thin launcher that shells out to the per-arch training scripts under
`runners/`, one subprocess per seed. The runners are **byte-identical copies of
experiment 16's** (themselves exp 10's: 06/07/08 + a `--weight-decay` /
`--lr-scheduler` knob, and NO early-stop callback -- a grokking probe must not
have one). It threads the **v2 tokenizers** through to each child: the 137-token
seq2seq tokenizer for vaswani / vaswani_rope, and the 138-token `<sot>`-augmented
one for gpt2_rope. Output dir, run name, wandb GROUP and tags all encode
(epochs, weight decay) so sweeping either knob lands in its own bucket.

DIFFERENCE FROM EXP 16 (only two knobs)
---------------------------------------
1. `--dataset` default -> the random-frame depth-2 dataset above (vs 16's
   held-out-depth split).
2. `--epochs` default -> 6, NOT 16's 50. The random train split (2,229,638 rows)
   is ~9x larger than 16's (~250k), so at batch 64 it is ~34,838 steps/epoch.
   6 epochs ~= 209k steps -- matched to 16's ~195k-step horizon so total
   optimization (and the wd pressure under constant LR) is comparable, and it
   fits one 2-day job. `--eval-steps`/`--save-steps` are step-based and unchanged,
   so eval resolution (~every 0.029 epoch -> ~209 points) and checkpoint count
   (~21) are comparable to 16's. Extend the horizon later with `--resume` +
   a larger `--epochs` (constant LR makes this clean).

LR schedule: defaults to 'constant_with_warmup' (flat LR after warmup), NOT
06/07/08's 'linear' decay-to-0 -- a decaying LR is ~0 by the late epochs where
grokking would occur, and constant LR lets the run be extended cleanly later.

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (RECOMMENDED) -- one job per architecture packs all seeds onto the
  single allocated GPU, throttled by --max-parallel. The models are tiny
  (~1.8M params, 4 layers) and underutilize an A6000, so 5 seeds share one GPU:
      sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch gpt2_rope

* Array -- 1 GPU per run. Lower wall-clock per run but uses 15 GPUs; prefer when
  the cluster is idle. If SLURM_ARRAY_TASK_ID is set, run exactly ONE (arch, seed)
  pair. With N seeds and 3 archs, use --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-14 slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py

LOCAL SMOKE TEST (2 seeds, tiny horizon, frequent eval/save, no wandb / no push)
--------------------------------------------------------------------------------
    uv run python experiments/18_multi_seed_random_depth2_grok/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 \\
        --eval-steps 50 --save-steps 50 --weight-decay 1.0 \\
        --max-parallel 2 --no-wandb

RESUMING (preemption / crash recovery / extending the horizon)
--------------------------------------------------------------
Pass `--resume` so each (arch, seed) auto-resumes from the latest checkpoint-* in
its OWN results dir (a run with none yet starts fresh, so it is safe from the
first submission). Best paired with the preemptable partition (packed form):
    sbatch slurm/run_preempt.sbatch \\
        experiments/18_multi_seed_random_depth2_grok/run.py --arch vaswani --resume
NB: the results dir encodes `ep<E>`, so a *larger* --epochs starts a fresh bucket
rather than extending in place -- set the full horizon up front (see README).

To copy: rename this file to run.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]            # experiments/18_.../ -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the LOCAL runner copy we drive (byte-identical to experiment 16's:
# 06/07/08 + a --weight-decay / --lr-scheduler knob, NO early-stop callback).
ARCH_RUNNERS: dict[str, Path] = {
    "vaswani":      EXP_DIR / "runners/vaswani_run.py",
    "vaswani_rope": EXP_DIR / "runners/vaswani_rope_run.py",
    "gpt2_rope":    EXP_DIR / "runners/gpt2_rope_run.py",
}
ARCHS = list(ARCH_RUNNERS)                # stable order for array-mode indexing

# wandb grouping: every seed of one (arch, epochs, wd) setting shares a run
# GROUP, so the wandb UI collapses them into one panel with a mean curve +
# min/max band across seeds. epochs AND weight decay fold into the group/tag
# names so a later longer-epoch or different-wd sweep does NOT merge in. Passed
# via env vars wandb.init() / the HF Trainer honor automatically.
WANDB_PREFIX = "18_multi_seed_random_depth2_grok"


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


def build_cmd(arch: str, seed: int, epochs: int, dataset: str, tokenizer: str,
              tokenizer_gpt2: str, wd: float, lr_scheduler: str, eval_steps: int,
              save_steps: int, max_eval_samples: int | None, no_wandb: bool, resume: bool):
    """Construct the argv for one (arch, seed) training and its output dir / run name."""
    suffix = tag(epochs, wd)
    out_dir = RESULTS_DIR / arch / suffix / f"seed_{seed}"
    run_name = f"{WANDB_PREFIX}_{arch}_{suffix}_seed_{seed}"
    # gpt2_rope needs the <sot>-augmented tokenizer; the two Vaswani enc-decs use
    # the plain seq2seq tokenizer.
    tok = tokenizer_gpt2 if arch == "gpt2_rope" else tokenizer
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--tokenizer", tok,
        "--weight-decay", str(wd),
        "--lr-scheduler", lr_scheduler,
        "--eval-steps", str(eval_steps),
        "--save-steps", str(save_steps),
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",                       # keep checkpoints local; do not create Hub repos
    ]
    if max_eval_samples is not None:
        cmd += ["--max-eval-samples", str(max_eval_samples)]
    if no_wandb:
        cmd.append("--no-wandb")
    if resume:
        # Bare flag -> each child auto-resumes from the latest checkpoint-* in
        # ITS OWN out_dir (above), so one --resume covers the whole sweep; a
        # child with no checkpoint yet falls back to a fresh run.
        cmd.append("--resume-from-checkpoint")
    return cmd, out_dir, run_name


def _launch(arch: str, seed: int, epochs: int, dataset: str, tokenizer: str,
            tokenizer_gpt2: str, wd: float, lr_scheduler: str, eval_steps: int,
            save_steps: int, max_eval_samples: int | None, no_wandb: bool, resume: bool):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(
        arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2, wd,
        lr_scheduler, eval_steps, save_steps, max_eval_samples, no_wandb, resume,
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


def run_packed(jobs, epochs, dataset, tokenizer, tokenizer_gpt2, wd, lr_scheduler,
               eval_steps, save_steps, max_eval_samples, no_wandb, resume, max_parallel) -> list[str]:
    """Run all (arch, seed) jobs as concurrent subprocesses, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            a, s = pending.pop(0)
            running.append(_launch(a, s, epochs, dataset, tokenizer, tokenizer_gpt2,
                                    wd, lr_scheduler, eval_steps, save_steps, max_eval_samples,
                                    no_wandb, resume))
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


def run_one(arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2, wd, lr_scheduler,
            eval_steps, save_steps, max_eval_samples, no_wandb, resume) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(
        arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2, wd, lr_scheduler,
        eval_steps, save_steps, max_eval_samples, no_wandb, resume,
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
    p.add_argument("--epochs", type=int, default=6,
                   help="Horizon to look for grokking. The random depth-2 train split has "
                        "2,229,638 rows (~34,838 steps/epoch at batch 64), so 6 epochs ~= 209k "
                        "steps -- matched to exp 16's ~195k-step horizon (16 used 50 epochs on a "
                        "~9x smaller split). Extend (constant LR makes this clean) via --resume "
                        "if you see clean memorization but flat held-out eval.")
    p.add_argument("--dataset",
                   default="kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random",
                   help="v2 grammar RANDOM-frame depth-2 split: train/val/test frame-disjoint "
                        "but all drawn from the same depth-{0,1,2} distribution (no structural "
                        "gap). The in-distribution control for exp 16's held-out-depth probe.")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="137-token v2 seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="138-token v2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--weight-decay", type=float, default=1.0,
                   help="AdamW L2 -- the main grokking lever. Default 1.0 (matches exp 16): "
                        "constant LR means wd carries all the regularization pressure. Sweep "
                        "0.1 / 1.0; each value lands in its own wandb group + results dir.")
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   help="HF lr_scheduler_type. Default 'constant_with_warmup' (flat LR after "
                        "warmup), NOT 06/07/08's 'linear' decay-to-0, so the late epochs where "
                        "grokking would appear aren't starved of LR and runs extend cleanly.")
    p.add_argument("--eval-steps", type=int, default=1000,
                   help="Step-based (unchanged from exp 16). On this dataset (~34,838 steps/epoch) "
                        "that is ~every 0.029 epoch -> ~209 eval points across 6 epochs: enough "
                        "resolution to see a transition without generation-eval dominating runtime.")
    p.add_argument("--save-steps", type=int, default=10000,
                   help="Step-based (unchanged from exp 16; 10x eval_steps) -> ~21 checkpoints "
                        "across 6 epochs. Must stay a multiple of --eval-steps (HF requires this "
                        "under load_best_model_at_end=True).")
    p.add_argument("--max-eval-samples", type=int, default=2500,
                   help="Threaded to each runner: evaluate on a fixed RANDOM subset of this many "
                        "val/test rows (shuffle seed 0) instead of the full split. Default 2500. "
                        "Full val/test are 278,704/278,706 rows here, and generation-eval over the "
                        "full split every --eval-steps would dwarf training (~13 min/eval x ~209 "
                        "evals ~= 47h, well past the walltime); 2500 rows -> ~7s/eval and a "
                        "+-1%% exact-match estimate. Pass a value >= the split size for full-set eval.")
    p.add_argument("--max-parallel", type=int, default=5,
                   help="Packed-mode concurrency on the single allocated GPU (ignored in array mode).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Pass through to each runner (skip wandb logging).")
    p.add_argument("--resume", action="store_true",
                   help="Resume each (arch, seed) from the latest checkpoint-* in its own "
                        "results dir (a run with no checkpoint yet starts fresh). Use to extend "
                        "a finished sweep: re-submit the SAME command with a larger --epochs and "
                        "--resume. Constant LR means no schedule recompute; resume restores "
                        "optimizer/scheduler/RNG from the checkpoint (the final best/ dir is "
                        "weights-only and is NOT used for resume).")
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
        sys.exit(run_one(arch, seed, args.epochs, args.dataset, args.tokenizer,
                         args.tokenizer_gpt2, args.weight_decay, args.lr_scheduler,
                         args.eval_steps, args.save_steps, args.max_eval_samples,
                         args.no_wandb, args.resume))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"epochs={args.epochs} wd={args.weight_decay} lr_scheduler={args.lr_scheduler} "
          f"eval_steps={args.eval_steps} save_steps={args.save_steps} "
          f"max_eval_samples={args.max_eval_samples} max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset, args.tokenizer,
                          args.tokenizer_gpt2, args.weight_decay, args.lr_scheduler,
                          args.eval_steps, args.save_steps, args.max_eval_samples,
                          args.no_wandb, args.resume, args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
