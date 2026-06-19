"""Small-model capacity sweep on the v2 grammar's *random* depth-2 split (experiment 20).

WHY
---
This is a **shrunk, single-epoch variant of experiment 18**. Exp 18 ran the
grokking-probe machinery (3 archs x 5 seeds, no early-stop, constant LR, wd=1.0)
on the v2 grammar's random-frame depth-2 split with the runners' *default*
4-layer / 4-head models over a 6-epoch horizon. 20 keeps the SAME dataset, archs,
seeds, tokenizers, wd/LR and eval cadence, but asks a different question: how do
**smaller** models (fewer layers / heads) fare on this in-distribution split after
just **one** epoch? It sweeps four capacity points per arch:

      config   layers  heads
      2x2      2       2
      2x1      2       1
      3x2      3       2
      3x1      3       1

  Dataset (unchanged from 18): kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random
  (train 2,229,638 / val 278,704 / test 278,706 rows; random 80/10/10 over frames,
   all drawn from the same depth-{0,1,2} distribution -- no structural holdout.)

DIFFERENCES FROM EXP 18 (only three knobs; runners are byte-identical copies)
----------------------------------------------------------------------------
1. `--epochs` default -> **1** (vs 18's 6). At batch 64 the train split is
   ~34,838 steps/epoch, so one epoch ~= 34,838 steps. This is a quick capacity
   probe, NOT a grokking horizon -- "grok" behavior (transition long past
   convergence) is not expected in a single epoch; we keep 18's no-early-stop /
   constant-LR / wd=1.0 config only so the comparison to 18 is apples-to-apples.
2. **Layer/head sweep.** The launcher now threads `--n-layers` / `--n-heads`
   (which all three runners already accept; 18 left them at the 4/4 default) and
   sweeps the four (layers, heads) configs above. d_model/d_ff/dropout are
   unchanged (128 / 512 / 0.1); heads of 1 and 2 both divide d_model=128.
3. **Eval, exactly as requested:** per-step generation-eval runs on a fixed
   RANDOM 2,500-row subset of val/test (`--max-eval-samples 2500`, same default as
   18) to keep training tractable; then -- already built into every runner -- a
   one-off `val_full` pass evaluates the (best) model on the **ENTIRE** 278,704-row
   validation set after the epoch completes. No runner edits were needed for this.

WHAT THIS FILE DOES (AND DOES NOT) DO
-------------------------------------
A thin launcher that shells out to the per-arch training scripts under `runners/`
(byte-identical copies of exp 18's), one subprocess per (arch, config, seed). It
threads the v2 tokenizers (137-token seq2seq for vaswani / vaswani_rope; 138-token
<sot>-augmented for gpt2_rope) and the per-config `--n-layers`/`--n-heads`. Output
dir, run name, wandb GROUP and tags all encode (layers, heads, epochs, weight
decay) so each capacity point lands in its own bucket and never merges with 18's.

SUBMISSION (RECOMMENDED): one packed job per (arch, config) -> 12 jobs, each
packing its 5 seeds on the single allocated GPU (throttled by --max-parallel).
The models are tiny and underutilize an A6000, so 5 seeds share one GPU:
    for cfg in 2x2 2x1 3x2 3x1; do
      for arch in vaswani vaswani_rope gpt2_rope; do
        sbatch slurm/run_gpu.sbatch \\
          experiments/20_multi_seed_random_depth2_small_models/run.py \\
          --arch $arch --config $cfg
      done
    done

ARRAY (alternative; 1 GPU per run, 60 tasks). If SLURM_ARRAY_TASK_ID is set, run
exactly ONE (arch, config, seed). With 3 archs x 4 configs x 5 seeds:
    sbatch --array=0-59 slurm/run_gpu.sbatch \\
        experiments/20_multi_seed_random_depth2_small_models/run.py
    # task -> (arch, config, seed): see array-mode block below for the indexing.

LOCAL SMOKE TEST (1 arch, 1 config, 2 seeds, frequent eval/save, no wandb / push)
---------------------------------------------------------------------------------
    uv run python experiments/20_multi_seed_random_depth2_small_models/run.py \\
        --arch vaswani --config 2x2 --seeds 42-43 \\
        --eval-steps 50 --save-steps 50 --max-parallel 2 --no-wandb

To copy: rename this file to run.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]            # experiments/20_.../ -> experiments -> repo root
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

# arch key -> the LOCAL runner copy we drive (byte-identical to experiment 18's).
ARCH_RUNNERS: dict[str, Path] = {
    "vaswani":      EXP_DIR / "runners/vaswani_run.py",
    "vaswani_rope": EXP_DIR / "runners/vaswani_rope_run.py",
    "gpt2_rope":    EXP_DIR / "runners/gpt2_rope_run.py",
}
ARCHS = list(ARCH_RUNNERS)                # stable order for array-mode indexing

# Capacity sweep: (n_layers, n_heads). Hardcoded (like ARCHS) so array-mode
# indexing is stable; --config selects one of these or "all" in packed mode.
CONFIGS: list[tuple[int, int]] = [(2, 2), (2, 1), (3, 2), (3, 1)]
CONFIG_KEYS = [f"{l}x{h}" for l, h in CONFIGS]   # "LxH" = layers x heads
CONFIG_BY_KEY = dict(zip(CONFIG_KEYS, CONFIGS))

# wandb grouping: every seed of one (arch, layers, heads, epochs, wd) setting
# shares a run GROUP, so the UI collapses them into one panel with a mean curve +
# min/max band. The capacity config folds into every name so 20's buckets never
# merge with 18's (or each other). Passed via env vars wandb/HF honor automatically.
WANDB_PREFIX = "20_multi_seed_random_depth2_small_models"


def parse_seeds(spec: str) -> list[int]:
    """'42-46' -> [42..46] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def tag(layers: int, heads: int, epochs: int, wd: float) -> str:
    """Shared (config, epochs, wd) suffix used in dir / run-name / group / tags."""
    return f"L{layers}H{heads}_ep{epochs}_wd{wd}"


def build_cmd(arch: str, layers: int, heads: int, seed: int, args):
    """Construct the argv for one (arch, config, seed) run and its out dir / name.

    Per-job knobs are (arch, layers, heads, seed); everything else is read off the
    parsed `args` namespace (the one structural change vs 18, which threaded every
    knob positionally -- with the added config dimension that became unwieldy).
    """
    suffix = tag(layers, heads, args.epochs, args.weight_decay)
    out_dir = RESULTS_DIR / arch / suffix / f"seed_{seed}"
    run_name = f"{WANDB_PREFIX}_{arch}_{suffix}_seed_{seed}"
    # gpt2_rope needs the <sot>-augmented tokenizer; the Vaswani enc-decs use the
    # plain seq2seq one.
    tok = args.tokenizer_gpt2 if arch == "gpt2_rope" else args.tokenizer
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--n-layers", str(layers),
        "--n-heads", str(heads),
        "--dataset", args.dataset,
        "--tokenizer", tok,
        "--weight-decay", str(args.weight_decay),
        "--lr-scheduler", args.lr_scheduler,
        "--eval-steps", str(args.eval_steps),
        "--save-steps", str(args.save_steps),
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",                       # keep checkpoints local; no Hub repos
    ]
    if args.max_eval_samples is not None:
        cmd += ["--max-eval-samples", str(args.max_eval_samples)]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.resume:
        # Bare flag -> each child auto-resumes from the latest checkpoint-* in ITS
        # OWN out_dir; a child with no checkpoint yet falls back to a fresh run.
        cmd.append("--resume-from-checkpoint")
    return cmd, out_dir, run_name


def _launch(arch: str, layers: int, heads: int, seed: int, args):
    """Start one training subprocess, streaming its output to a per-run log file."""
    cmd, out_dir, run_name = build_cmd(arch, layers, heads, seed, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Group this (arch, config, epochs, wd)'s seeds together in wandb; tag for
    # filtering. Harmless when --no-wandb (the child sets report_to="none").
    suffix = tag(layers, heads, args.epochs, args.weight_decay)
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{arch}_{suffix}"
    env["WANDB_TAGS"] = (f"{WANDB_PREFIX},{arch},L{layers}H{heads},seed_{seed},"
                         f"ep{args.epochs},wd{args.weight_decay}")
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, args) -> list[str]:
    """Run all (arch, layers, heads, seed) jobs concurrently, max_parallel at a time."""
    pending = list(jobs)
    running: list[tuple] = []              # (proc, run_name, log handle)
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            a, l, h, s = pending.pop(0)
            running.append(_launch(a, l, h, s, args))
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


def run_one(arch, layers, heads, seed, args) -> int:
    """Run a single training to completion (array mode). Returns its exit code."""
    proc, run_name, log = _launch(arch, layers, heads, seed, args)
    rc = proc.wait()
    log.close()
    print(f"[done ] {run_name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=[*ARCHS, "all"], default="all",
                   help="Which architecture to run in packed mode (ignored in array mode, "
                        "which always sweeps all three).")
    p.add_argument("--config", choices=[*CONFIG_KEYS, "all"], default="all",
                   help="Which capacity config 'LxH' (layers x heads) to run in packed mode. "
                        "Default 'all' packs every config; pass one (e.g. 2x2) for the "
                        "recommended one-job-per-(arch,config) submission. Ignored in array mode.")
    p.add_argument("--seeds", default="42-46",
                   help="Inclusive range 'A-B' or comma list. Default 42-46 (5 seeds).")
    p.add_argument("--epochs", type=int, default=1,
                   help="Default 1 (vs exp 18's 6). One epoch ~= 34,838 steps at batch 64 on "
                        "this ~2.23M-row train split -- a quick capacity probe, not a grokking "
                        "horizon. The no-early-stop / constant-LR / wd config is kept only so "
                        "the comparison to exp 18 stays apples-to-apples.")
    p.add_argument("--dataset",
                   default="kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random",
                   help="v2 grammar RANDOM-frame depth-2 split (unchanged from exp 18): "
                        "train/val/test frame-disjoint but all from the same depth-{0,1,2} "
                        "distribution -- no structural gap.")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="137-token v2 seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="138-token v2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--weight-decay", type=float, default=1.0,
                   help="AdamW L2. Default 1.0 (matches exp 18). Each value lands in its own "
                        "wandb group + results dir.")
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   help="HF lr_scheduler_type. Default 'constant_with_warmup' (matches exp 18).")
    p.add_argument("--eval-steps", type=int, default=1000,
                   help="Step-based (unchanged from exp 18). At ~34,838 steps/epoch that is "
                        "~34 eval points across the single epoch.")
    p.add_argument("--save-steps", type=int, default=10000,
                   help="Step-based (unchanged from exp 18; 10x eval_steps) -> ~3 checkpoints "
                        "across one epoch. Must stay a multiple of --eval-steps (HF requires "
                        "this under load_best_model_at_end=True).")
    p.add_argument("--max-eval-samples", type=int, default=2500,
                   help="Per-step generation-eval runs on a fixed RANDOM subset of this many "
                        "val/test rows (shuffle seed 0); default 2500 (same as exp 18). The FULL "
                        "278,704-row validation set is evaluated once after the epoch by each "
                        "runner's built-in `val_full` pass. Pass a value >= the split size to "
                        "make per-step eval full-set too.")
    p.add_argument("--max-parallel", type=int, default=5,
                   help="Packed-mode concurrency on the single allocated GPU (ignored in array mode).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Pass through to each runner (skip wandb logging).")
    p.add_argument("--resume", action="store_true",
                   help="Resume each (arch, config, seed) from the latest checkpoint-* in its "
                        "own results dir (a run with no checkpoint yet starts fresh).")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)

    if args.save_steps % args.eval_steps != 0:
        sys.exit(f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
                 f"({args.eval_steps}); HF requires this when load_best_model_at_end=True.")

    # --- Array mode: SLURM_ARRAY_TASK_ID selects exactly one (arch, config, seed) ---
    # Layout: task = arch_i*(n_configs*n_seeds) + config_i*n_seeds + seed_i.
    task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_env is not None:
        task = int(task_env)
        n_seeds = len(seeds)
        n_cfg = len(CONFIGS)
        total = len(ARCHS) * n_cfg * n_seeds
        if task >= total:
            sys.exit(f"SLURM_ARRAY_TASK_ID={task} out of range for {len(ARCHS)} archs x "
                     f"{n_cfg} configs x {n_seeds} seeds (use --array=0-{total - 1}).")
        arch = ARCHS[task // (n_cfg * n_seeds)]
        rem = task % (n_cfg * n_seeds)
        layers, heads = CONFIGS[rem // n_seeds]
        seed = seeds[rem % n_seeds]
        print(f"[array] task {task} -> arch={arch} config=L{layers}H{heads} seed={seed}", flush=True)
        sys.exit(run_one(arch, layers, heads, seed, args))

    # --- Packed mode ----------------------------------------------------------
    archs = ARCHS if args.arch == "all" else [args.arch]
    configs = CONFIGS if args.config == "all" else [CONFIG_BY_KEY[args.config]]
    jobs = [(a, l, h, s) for a in archs for (l, h) in configs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} "
          f"configs={[f'L{l}H{h}' for l, h in configs]} seeds={seeds} "
          f"epochs={args.epochs} wd={args.weight_decay} lr_scheduler={args.lr_scheduler} "
          f"eval_steps={args.eval_steps} save_steps={args.save_steps} "
          f"max_eval_samples={args.max_eval_samples} max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
