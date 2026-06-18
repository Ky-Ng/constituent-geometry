"""Multi-seed sweep on the v2 grammar depth-2 adjunction dataset (experiment 15, v2).

Updated from the original run.py to:
* Use local training scripts (train_vaswani.py / train_vaswani_rope.py /
  train_gpt2_rope.py) instead of delegating to experiments 06-08, so the
  ThresholdStopCallback and v2 tokenizer paths can be passed through.
* Add --stop-threshold (default 0.98): stop 1 epoch after eval_exact_match
  hits the threshold.  Pass None to disable and train for all --epochs.
* Add --tokenizer (seq2seq, for vaswani / vaswani_rope) and --tokenizer-gpt2
  (for gpt2_rope which needs the <sot> token).
* Updated --dataset default to the adjunction dataset.

TWO SUBMISSION MODES (same file)
---------------------------------
* Packed (default) -- one job per architecture packs all seeds onto the single
  allocated GPU, throttled by --max-parallel:
      sbatch slurm/run_gpu.sbatch experiments/15_multi_seed_grammar_v2_depth_2_random/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/15_multi_seed_grammar_v2_depth_2_random/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/15_multi_seed_grammar_v2_depth_2_random/run.py --arch gpt2_rope

* Array (1 GPU per run) -- if SLURM_ARRAY_TASK_ID is set, run exactly ONE
  (arch, seed) pair. With N seeds and 3 archs, use --array=0-(3N-1):
      sbatch --array=0-29 slurm/run_gpu.sbatch experiments/15_multi_seed_grammar_v2_depth_2_random/run.py

LOCAL SMOKE TEST (2 seeds, 1 epoch, no wandb)
---------------------------------------------
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb

To copy: rename this file to run.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

ARCH_RUNNERS: dict[str, Path] = {
    "vaswani":      EXP_DIR / "train_vaswani.py",
    "vaswani_rope": EXP_DIR / "train_vaswani_rope.py",
    "gpt2_rope":    EXP_DIR / "train_gpt2_rope.py",
}
ARCHS = list(ARCH_RUNNERS)

WANDB_PREFIX = "15_multi_seed_grammar_v2_depth_2_random"
# Disambiguates this batch (v2 adjunction tokenizer + early stopping) from the
# original exp-15 runs that shared the same run names. Threaded into the wandb
# run name, run group, and tags.
VARIANT = "v2"


def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51] (inclusive); '42,44,46' -> [42, 44, 46]."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def build_cmd(
    arch: str,
    seed: int,
    epochs: int,
    dataset: str,
    tokenizer: str,
    tokenizer_gpt2: str,
    stop_threshold: float | None,
    no_wandb: bool,
):
    out_dir = RESULTS_DIR / arch / f"seed_{seed}"
    run_name = f"{WANDB_PREFIX}_{VARIANT}_{arch}_seed_{seed}"
    tok = tokenizer_gpt2 if arch == "gpt2_rope" else tokenizer
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--tokenizer", tok,
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",
    ]
    if stop_threshold is not None:
        cmd += ["--stop-threshold", str(stop_threshold)]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd, out_dir, run_name


def _launch(arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2,
            stop_threshold, no_wandb):
    cmd, out_dir, run_name = build_cmd(
        arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2,
        stop_threshold, no_wandb,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{VARIANT}_{arch}"
    env["WANDB_TAGS"] = f"{WANDB_PREFIX},{VARIANT},{arch},seed_{seed}"
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, epochs, dataset, tokenizer, tokenizer_gpt2,
               stop_threshold, no_wandb, max_parallel) -> list[str]:
    pending = list(jobs)
    running: list[tuple] = []
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            arch, seed = pending.pop(0)
            running.append(_launch(arch, seed, epochs, dataset, tokenizer,
                                   tokenizer_gpt2, stop_threshold, no_wandb))
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


def run_one(arch, seed, epochs, dataset, tokenizer, tokenizer_gpt2,
            stop_threshold, no_wandb) -> int:
    proc, run_name, log = _launch(arch, seed, epochs, dataset, tokenizer,
                                  tokenizer_gpt2, stop_threshold, no_wandb)
    rc = proc.wait()
    log.close()
    print(f"[done ] {run_name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=[*ARCHS, "all"], default="all")
    p.add_argument("--seeds", default="42-51",
                   help="Inclusive range 'A-B' or comma list. Default 42-51 (10 seeds).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-frames-d2-k100-random")
    p.add_argument("--tokenizer",
                   default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="Seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2",
                   default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="GPT2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--stop-threshold", type=float, default=0.98,
                   help="Stop 1 epoch after eval_exact_match >= this. "
                        "Set to 0 or omit to disable.")
    p.add_argument("--max-parallel", type=int, default=10)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    stop_threshold = args.stop_threshold if args.stop_threshold else None

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
                         args.tokenizer_gpt2, stop_threshold, args.no_wandb))

    archs = ARCHS if args.arch == "all" else [args.arch]
    jobs = [(a, s) for a in archs for s in seeds]
    print(f"[packed] {len(jobs)} run(s): archs={archs} seeds={seeds} "
          f"max_parallel={args.max_parallel}", flush=True)
    failures = run_packed(jobs, args.epochs, args.dataset, args.tokenizer,
                          args.tokenizer_gpt2, stop_threshold, args.no_wandb,
                          args.max_parallel)
    if failures:
        sys.exit(f"\n{len(failures)} run(s) FAILED: {failures}")
    print("\nAll runs completed.", flush=True)


if __name__ == "__main__":
    main()
