"""Multi-seed sweep on the v2 grammar depth-1 dataset (experiment 14).

Mirrors experiment 09 exactly, except the dataset is the v2 grammar
(`kylelovesllms/hi-hf-v2-frames-d1-k100`) instead of the v1 depth-3 dataset.
All three architectures (vaswani, vaswani_rope, gpt2_rope) are run across
10 seeds (42-51, same as 09) so results are directly comparable.

TWO SUBMISSION MODES (same file)
---------------------------------
* Packed (default) -- one job per architecture packs all seeds onto the single
  allocated GPU, throttled by --max-parallel:
      sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch vaswani
      sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch vaswani_rope
      sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch gpt2_rope

* Array (1 GPU per run) -- if SLURM_ARRAY_TASK_ID is set, run exactly ONE
  (arch, seed) pair. With N seeds and 3 archs, use --array=0-(3N-1):
      sbatch --array=0-29 slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py

LOCAL SMOKE TEST (2 seeds, 1 epoch, no wandb)
---------------------------------------------
    uv run python experiments/14_multi_seed_grammar_v2_depth_1/run.py \\
        --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
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
    "vaswani":      REPO_ROOT / "experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py",
    "vaswani_rope": REPO_ROOT / "experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py",
    "gpt2_rope":    REPO_ROOT / "experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py",
}
ARCHS = list(ARCH_RUNNERS)

WANDB_PREFIX = "14_multi_seed_grammar_v2_depth_1"


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
    out_dir = RESULTS_DIR / arch / f"seed_{seed}"
    run_name = f"14_multi_seed_grammar_v2_depth_1_{arch}_seed_{seed}"
    cmd = [
        sys.executable, str(ARCH_RUNNERS[arch]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--no-push",
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd, out_dir, run_name


def _launch(arch: str, seed: int, epochs: int, dataset: str, no_wandb: bool):
    cmd, out_dir, run_name = build_cmd(arch, seed, epochs, dataset, no_wandb)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"{WANDB_PREFIX}_{arch}"
    env["WANDB_TAGS"] = f"{WANDB_PREFIX},{arch},seed_{seed}"
    log = open(LOGS_DIR / f"{run_name}.log", "w")
    print(f"[start] {run_name}  (log: logs/{run_name}.log)", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, env=env)
    return proc, run_name, log


def run_packed(jobs, epochs, dataset, no_wandb, max_parallel) -> list[str]:
    pending = list(jobs)
    running: list[tuple] = []
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
    proc, run_name, log = _launch(arch, seed, epochs, dataset, no_wandb)
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
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-frames-d1-k100")
    p.add_argument("--max-parallel", type=int, default=10)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)

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
