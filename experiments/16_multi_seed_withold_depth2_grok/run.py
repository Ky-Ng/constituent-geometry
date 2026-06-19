"""Seed-sweep GROKKING probe on the v2 grammar's *held-out-depth-2* split (experiment 16).

WHY
---
Experiment 10 ran exactly this probe -- 3 archs x seeds, long horizon, a
`--weight-decay` lever, constant LR -- on the *v1* grammar's held-out-depth
split, and at wd=0.1 it did NOT grok: every run memorized (`train_loss -> 0`)
and then *overfit* (held-out `eval_loss` climbed to 3-6, test exact-match 0.0).
16 reruns the same probe on the **v2 grammar** (relative clauses + adjunction,
the 137/138-token v2 tokenizer) and its **depth-withhold-2** split: train sees
only shallow trees; val/test are DEEPER, held-out structures. The question is
whether deeper-depth compositional generalization is recovered by long training
+ weight decay on the richer v2 grammar.

  Grokking signature here: `train_loss` collapses early (memorization of the
  shallow train frames) while held-out `eval_exact_match` stays low, then -- if
  grokking occurs -- rises sharply much later.

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
A thin launcher that shells out to the per-arch training scripts under
`runners/`, one subprocess per seed. The runners are **byte-identical copies of
experiment 10's** runners (themselves 06/07/08 + a `--weight-decay` and
`--lr-scheduler` knob, and -- importantly -- NO early-stop callback, which a
grokking probe must not have). This launcher differs from 10's only in:
  - It threads the **v2 tokenizer** through to each child (`--tokenizer`):
    the seq2seq 137-token tokenizer for vaswani / vaswani_rope, and the
    138-token `<sot>`-augmented one for gpt2_rope. (10 relied on the runners'
    v1 tokenizer defaults; the v2 dataset needs the v2 vocab.)
  - Defaults retuned for the v2 withhold-depth-2 dataset, which is ~60x larger
    than 10's (train ~250k vs ~4k -> ~3,906 steps/epoch at batch 64). So the
    epoch budget and the eval/save cadence are set in *steps* appropriate to
    that size (see `--epochs` / `--eval-steps` / `--save-steps` help).
  - Output dir, run name, wandb GROUP, and tags all encode (epochs, weight
    decay) so sweeping either knob lands in its own bucket instead of merging.

LR schedule: defaults to 'constant_with_warmup' (flat LR after warmup), NOT
06/07/08's 'linear' decay-to-0 -- a decaying LR is ~0 by the late epochs where
grokking would occur, and constant LR lets the run be extended cleanly later.
The `--weight-decay` knob supplies the regularization pressure LR decay would.

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (RECOMMENDED for 16) -- one job per architecture packs all seeds onto
  the single allocated GPU, throttled by --max-parallel. The models are tiny
  (~1.8M params, 4 layers) and underutilize an A6000, so 5 seeds share one GPU
  efficiently -- 3 GPUs total instead of 15, friendlier to the shared cluster:
      sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch gpt2_rope

* Array -- 1 GPU per run. Lower wall-clock per run but uses 15 GPUs; prefer when
  the cluster is idle and you want results fastest. If SLURM_ARRAY_TASK_ID is
  set, run exactly ONE (arch, seed) pair. With N seeds and 3 archs, use
  --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-14 slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py

LOCAL SMOKE TEST (2 seeds, ~tiny horizon, frequent eval/save, no wandb / no push)
--------------------------------------------------------------------------------
    uv run python experiments/16_multi_seed_withold_depth2_grok/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 \\
        --eval-steps 50 --save-steps 50 --weight-decay 1.0 \\
        --max-parallel 2 --no-wandb

RESUMING (preemption / crash recovery)
--------------------------------------
Pass `--resume` so each (arch, seed) auto-resumes from the latest checkpoint-* in
its OWN results dir (a run with none yet starts fresh, so it is safe from the
first submission). Best paired with the preemptable partition (packed form):
    sbatch slurm/run_preempt.sbatch \\
        experiments/16_multi_seed_withold_depth2_grok/run.py --arch vaswani --resume
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
REPO_ROOT = EXP_DIR.parents[1]            # experiments/16_.../ -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the LOCAL runner copy we drive (byte-identical to experiment 10's:
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
WANDB_PREFIX = "16_multi_seed_withold_depth2_grok"


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
              save_steps: int, no_wandb: bool, resume: bool):
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
            save_steps: int, no_wandb: bool, resume: bool):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(
        arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2, wd,
        lr_scheduler, eval_steps, save_steps, no_wandb, resume,
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
               eval_steps, save_steps, no_wandb, resume, max_parallel) -> list[str]:
    """Run all (arch, seed) jobs as concurrent subprocesses, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            a, s = pending.pop(0)
            running.append(_launch(a, s, epochs, dataset, tokenizer, tokenizer_gpt2,
                                    wd, lr_scheduler, eval_steps, save_steps, no_wandb,
                                    resume))
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
            eval_steps, save_steps, no_wandb, resume) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(
        arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2, wd, lr_scheduler,
        eval_steps, save_steps, no_wandb, resume,
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
    p.add_argument("--epochs", type=int, default=50,
                   help="Horizon to look for grokking. The v2 withhold-depth-2 train split has "
                        "~250k rows (~3,906 steps/epoch at batch 64), so 50 epochs ~= 195k steps "
                        "-- comparable to exp 10's total step count. Extend (constant LR makes "
                        "this clean) if you see clean memorization but flat held-out eval.")
    p.add_argument("--dataset",
                   default="kylelovesllms/hi-hf-v2-frames-depth_withhold2_k_train100_k_eval5_eval_frame_cap500",
                   help="v2 grammar held-out-DEPTH split: train = shallow trees, "
                        "val/test = deeper held-out structures.")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="137-token v2 seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="138-token v2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--weight-decay", type=float, default=1.0,
                   help="AdamW L2 -- the main grokking lever. Default 1.0: exp 10 showed wd=0.1 "
                        "fails to grok on the analogous v1 task, and constant LR means wd carries "
                        "all the regularization pressure. Sweep 0.1 / 1.0; each value lands in its "
                        "own wandb group + results dir.")
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   help="HF lr_scheduler_type. Default 'constant_with_warmup' (flat LR after "
                        "warmup), NOT 06/07/08's 'linear' decay-to-0, so the late epochs where "
                        "grokking would appear aren't starved of LR and runs extend cleanly.")
    p.add_argument("--eval-steps", type=int, default=1000,
                   help="~every 0.26 epoch on this dataset (~3,906 steps/epoch) -> ~195 eval "
                        "points across 50 epochs: enough resolution to see a grok transition "
                        "without generation-eval dominating runtime.")
    p.add_argument("--save-steps", type=int, default=10000,
                   help="Kept LARGE (10x eval_steps) so 50 epochs writes ~20 checkpoints/run "
                        "instead of ~195. Must stay a multiple of --eval-steps (HF requires this "
                        "under load_best_model_at_end=True).")
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
                         args.eval_steps, args.save_steps, args.no_wandb, args.resume))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"epochs={args.epochs} wd={args.weight_decay} lr_scheduler={args.lr_scheduler} "
          f"eval_steps={args.eval_steps} save_steps={args.save_steps} "
          f"max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset, args.tokenizer,
                          args.tokenizer_gpt2, args.weight_decay, args.lr_scheduler,
                          args.eval_steps, args.save_steps, args.no_wandb, args.resume,
                          args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
