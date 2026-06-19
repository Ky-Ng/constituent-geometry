"""Seed-sweep GROKKING probe on the v2 grammar's *RC-on-subject held-out* split (experiment 19).

WHY
---
This is a **structural-generalization** probe in the spirit of experiment 16, but
the held-out axis is *which NP carries an adjunct* rather than *depth*. It asks
whether a relative clause -- a post-nominal NP adjunct -- learned only on OBJECT
noun phrases transfers to the SUBJECT position.

  Dataset: kylelovesllms/hi-hf-v2-frames-object-rc-only-depth2
    train (3,936): subject_branching=False, subject_mod='' EVERYWHERE -- subjects
      are always atomic (proper name or bare D N); every adjective / relative
      clause lives on an OBJECT. depth {0,1,2} (mostly 2).
    val/test (300 each): subject_branching=True, subject_mod in {rel, adj+rel} --
      the SUBJECT now carries a relative clause (an NP adjunct). depth 2,
      frame-disjoint (100 frames each).

  So train and val/test differ by a genuine STRUCTURAL gap (RC/adjunct on the
  subject is never seen in training), exactly the kind of gap a grokking probe
  needs -- this is the position-transfer analog of 16's depth gap, not 18's
  in-distribution control.

  Grokking signature here: `train_loss` collapses early (memorization of the
  object-only-adjunct train frames) while held-out `eval_exact_match` on the
  subject-RC structures stays low, then -- if grokking occurs -- rises sharply
  much later. (This is the depth-2 incarnation of exp 17's subject-position
  hypothesis, run with the no-early-stop / wd=1.0 / constant-LR grokking config.)

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
A thin launcher that shells out to the per-arch training scripts under
`runners/`, one subprocess per seed. The runners are **byte-identical copies of
experiment 16/18's** (themselves exp 10's: 06/07/08 + a `--weight-decay` /
`--lr-scheduler` knob, and NO early-stop callback -- a grokking probe must not
have one). It threads the **v2 tokenizers** through to each child: the 137-token
seq2seq tokenizer for vaswani / vaswani_rope, and the 138-token `<sot>`-augmented
one for gpt2_rope. The v2 grammar (and thus the vocab) is identical to 15/16/18,
so the tokenizers are reused unchanged. Output dir, run name, wandb GROUP and
tags all encode (epochs, weight decay) so sweeping either knob lands in its own
bucket.

DIFFERENCE FROM EXP 16/18 (only two knobs)
------------------------------------------
1. `--dataset` default -> the RC-on-subject held-out depth-2 dataset above (vs
   16's held-out-DEPTH split and 18's random-frame split).
2. `--epochs` default -> 3000, NOT 16's 50 or 18's 6. This train split is TINY
   (3,936 rows -> ~62 steps/epoch at batch 64), the same scale as exp 10's
   grokking probe (~66 steps/epoch). So the horizon is matched in *epochs* to
   exp 10 (3000): 3000 x ~62 ~= 186k steps -- squarely in the ~195k-step grokking
   horizon of exp 10/16/18. `--eval-steps`/`--save-steps` are step-based and
   unchanged, so ~186 eval points (~every 16 epochs) and ~18 checkpoints across
   the run -- comparable resolution to 16/18. Extend later with `--resume` + a
   larger `--epochs` (constant LR makes this clean).

LR schedule: defaults to 'constant_with_warmup' (flat LR after warmup), NOT
06/07/08's 'linear' decay-to-0 -- a decaying LR is ~0 by the late epochs where
grokking would occur, and constant LR lets the run be extended cleanly later.

TWO SUBMISSION MODES (same file)
--------------------------------
* Packed (RECOMMENDED) -- one job per architecture packs all seeds onto the
  single allocated GPU, throttled by --max-parallel. The models are tiny
  (~1.8M params, 4 layers) and underutilize an A6000, so 5 seeds share one GPU:
      sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch gpt2_rope

* Array -- 1 GPU per run. Lower wall-clock per run but uses 15 GPUs; prefer when
  the cluster is idle. If SLURM_ARRAY_TASK_ID is set, run exactly ONE (arch, seed)
  pair. With N seeds and 3 archs, use --array=0-(3N-1):
      arch = ARCHS[task // N];  seed = seeds[task % N]
      sbatch --array=0-14 slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py

LOCAL SMOKE TEST (2 seeds, tiny horizon, frequent eval/save, no wandb / no push)
--------------------------------------------------------------------------------
    uv run python experiments/19_NP_adjunct_generalization/run.py \\
        --arch vaswani --seeds 42-43 --epochs 5 \\
        --eval-steps 50 --save-steps 50 --weight-decay 1.0 \\
        --max-parallel 2 --no-wandb

RESUMING (preemption / crash recovery / extending the horizon)
--------------------------------------------------------------
Pass `--resume` so each (arch, seed) auto-resumes from the latest checkpoint-* in
its OWN results dir (a run with none yet starts fresh, so it is safe from the
first submission). Best paired with the preemptable partition (packed form):
    sbatch slurm/run_preempt.sbatch \\
        experiments/19_NP_adjunct_generalization/run.py --arch vaswani --resume
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
REPO_ROOT = EXP_DIR.parents[1]            # experiments/19_.../ -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the LOCAL runner copy we drive (byte-identical to experiment 16/18's:
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
WANDB_PREFIX = "19_NP_adjunct_generalization"


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
    p.add_argument("--epochs", type=int, default=3000,
                   help="Horizon to look for grokking. This RC-on-subject train split is TINY "
                        "(3,936 rows -> ~62 steps/epoch at batch 64), the same scale as exp 10's "
                        "grokking probe (~66 steps/epoch). So the horizon is matched in EPOCHS to "
                        "exp 10 (3000): 3000 x ~62 ~= 186k steps -- in the ~195k-step grokking "
                        "horizon of exp 10/16/18. Extend (constant LR makes this clean) via "
                        "--resume if you see clean memorization but flat held-out eval.")
    p.add_argument("--dataset",
                   default="kylelovesllms/hi-hf-v2-frames-object-rc-only-depth2",
                   help="v2 grammar RC-on-subject HELD-OUT depth-2 split: train = subjects always "
                        "atomic (every adjective/relative clause is on an OBJECT); val/test = the "
                        "SUBJECT carries a relative clause (NP adjunct). A real structural gap "
                        "(object->subject transfer of NP adjuncts), the position analog of 16's "
                        "depth gap.")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="137-token v2 seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="138-token v2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--weight-decay", type=float, default=1.0,
                   help="AdamW L2 -- the main grokking lever. Default 1.0 (matches exp 16/18): "
                        "constant LR means wd carries all the regularization pressure. Sweep "
                        "0.1 / 1.0; each value lands in its own wandb group + results dir.")
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   help="HF lr_scheduler_type. Default 'constant_with_warmup' (flat LR after "
                        "warmup), NOT 06/07/08's 'linear' decay-to-0, so the late epochs where "
                        "grokking would appear aren't starved of LR and runs extend cleanly.")
    p.add_argument("--eval-steps", type=int, default=1000,
                   help="Step-based (unchanged from 16/18). On this dataset (~62 steps/epoch) "
                        "that is ~every 16 epochs -> ~186 eval points across 3000 epochs: enough "
                        "resolution to see a transition without generation-eval dominating runtime.")
    p.add_argument("--save-steps", type=int, default=10000,
                   help="Step-based (unchanged from 16/18; 10x eval_steps) -> ~18 checkpoints "
                        "across 3000 epochs. Must stay a multiple of --eval-steps (HF requires "
                        "this under load_best_model_at_end=True).")
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
