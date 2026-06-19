"""Post-hoc FULL-validation-set evaluation for experiment 20 (separate from training).

WHY THIS EXISTS
---------------
Exp 20's runners now do ONLY capped (`--max-eval-samples 2500`) eval during
training, so each of the 60 runs finishes fast across all seeds. The expensive
full-set generation eval that exp 18 ran inline at the end of every run was
*moved here*, so you pay it once, on purpose, when you actually want the headline
number. This script loads each run's saved `best/` checkpoint and scores it on the
**ENTIRE** validation split (278,704 rows) -- and optionally the full test split.

SINGLE SOURCE OF TRUTH FOR EVAL LOGIC
-------------------------------------
It does NOT re-implement eval. It imports the exact building blocks the runners
use, from the runner files in this experiment's `runners/`:
  * seq2seq archs (vaswani / vaswani_rope): `make_tokenize_fn` (labels drop the
    leading <bos>), then a `Seq2SeqTrainer` with `predict_with_generate`.
  * decoder-only (gpt2_rope): `make_tokenize_fn` + `CausalLMCollator` +
    `GenEvalTrainer` (prompt = `[<bos>] hi <sot>`, greedy generate, strip prompt).
The metric is the same `exact_match` (string match after `skip_special_tokens`).
So a number here is directly comparable to the per-step `eval_exact_match` curve.

WHAT IT LOADS
-------------
Each run wrote its (best-by-capped-val-EM) checkpoint to
  results/<arch>/L<L>H<H>_ep<E>_wd<WD>/seed_<S>/best/
This script resolves that path for every selected (arch, config, seed), loads the
model with the matching custom class via `from_pretrained`, and evaluates it.
The (arch, config, seed) selection + the array-task layout MIRROR `run.py`, so the
same `--arch/--config/--seeds/--epochs/--weight-decay` pick out the same runs.

EFFICIENCY
----------
Tokenizing the full 278k-row split is the costly setup, and it is **arch-specific
but seed-independent**, so in-process (packed) mode tokenizes each split ONCE per
arch and reuses it across that arch's configs/seeds. Array mode does one
(arch, config, seed) per task -- no reuse, but it spreads the generation cost
across GPUs. Generation over the full split is heavy (~10+ min/model for the
Vaswani archs); prefer array mode for the full 60-run sweep.

OUTPUT
------
One JSON per run at `results/full_eval/<arch>_L<L>H<H>_ep<E>_wd<WD>_seed_<S>.json`
(safe for parallel array writes -- no shared file). In-process mode also prints a
summary table. `--summarize` scans those JSONs and writes `results/full_eval/
summary.csv` WITHOUT running any eval -- run it once after an array job finishes.

RUN
---
Single arch+config, all 5 seeds, full validation (in-process):
    uv run python experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py \\
        --arch vaswani --config 2x2

Full 60-run sweep as an array (1 GPU per run); then aggregate:
    sbatch --array=0-59 slurm/run_gpu.sbatch \\
        experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py
    uv run python experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py --summarize

Quick sanity (cap to 500 rows, no GPU needed if tiny):
    uv run python .../run_full_evaluate.py --arch vaswani --config 2x2 --seeds 42 \\
        --max-eval-samples 500

To copy: rename this file to run_full_evaluate.py.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RUNNERS_DIR = EXP_DIR / "runners"
DEFAULT_OUT = RESULTS_DIR / "full_eval"

# --- selection space: MUST match run.py so the same flags pick the same runs ---
ARCHS = ["vaswani", "vaswani_rope", "gpt2_rope"]
CONFIGS: list[tuple[int, int]] = [(2, 2), (2, 1), (3, 2), (3, 1)]
CONFIG_KEYS = [f"{l}x{h}" for l, h in CONFIGS]
CONFIG_BY_KEY = dict(zip(CONFIG_KEYS, CONFIGS))
# per-arch generation/truncation length = the runner default (launcher passes none):
#   vaswani / vaswani_rope -> 64 ; gpt2_rope -> 128 (concatenated seq is ~2x longer).
ARCH_MAX_LENGTH = {"vaswani": 64, "vaswani_rope": 64, "gpt2_rope": 128}


def parse_seeds(spec: str) -> list[int]:
    """'42-46' -> [42..46] (inclusive); '42,44,46' -> [42, 44, 46]. (Same as run.py.)"""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def tag(layers: int, heads: int, epochs: int, wd: float) -> str:
    """(config, epochs, wd) dir suffix -- identical to run.py's tag()."""
    return f"L{layers}H{heads}_ep{epochs}_wd{wd}"


def best_dir(arch: str, layers: int, heads: int, epochs: int, wd: float, seed: int,
             results_dir: Path) -> Path:
    """results/<arch>/L<L>H<H>_ep<E>_wd<WD>/seed_<S>/best (where the runner saved it)."""
    return results_dir / arch / tag(layers, heads, epochs, wd) / f"seed_{seed}" / "best"


def out_json_path(arch, layers, heads, epochs, wd, seed, out_dir: Path) -> Path:
    return out_dir / f"{arch}_{tag(layers, heads, epochs, wd)}_seed_{seed}.json"


def _load_runner(name: str):
    """Import a runner module from this experiment's runners/ by file path.

    Importing only runs its top-level imports (the runners guard main() under
    __name__ == '__main__'), so this is side-effect free. We reuse its eval
    building blocks rather than copy them, keeping the runners the source of truth.
    """
    path = RUNNERS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp20_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model_class(arch: str):
    """The custom model class to load each arch's best/ checkpoint with."""
    if arch == "vaswani":
        from architecture.modeling_vaswani import VaswaniForConditionalGeneration as C
    elif arch == "vaswani_rope":
        from architecture.modeling_vaswani_rope import VaswaniRoPEForConditionalGeneration as C
    elif arch == "gpt2_rope":
        from architecture.modeling_gpt2_rope import GPT2RoPEForCausalLM as C
    else:
        raise ValueError(f"unknown arch {arch}")
    return C


def make_compute_metrics(tok):
    """exact_match over decoded strings. Mirrors the runners: replace -100 with the
    pad id (a no-op for the gpt2 path, whose tensors are pad-padded), decode with
    skip_special_tokens, compare stripped strings."""
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_txt = tok.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tok.batch_decode(labels, skip_special_tokens=True)
        return {"exact_match": float(
            np.mean([p.strip() == r.strip() for p, r in zip(pred_txt, ref_txt)])
        )}
    return compute_metrics


# =============================================================================
# Per-arch eval engines: tokenize each requested split ONCE (shared across the
# arch's configs/seeds), then evaluate a given best/ checkpoint on each split.
# =============================================================================
class _BaseEngine:
    def __init__(self, arch, tok, tokenized, max_length, batch_size):
        self.arch = arch
        self.tok = tok
        self.tokenized = tokenized          # {split_name: tokenized Dataset}
        self.max_length = max_length
        self.batch_size = batch_size
        self.compute_metrics = make_compute_metrics(tok)

    def evaluate_model(self, ckpt: Path) -> dict:
        raise NotImplementedError


class Seq2SeqEngine(_BaseEngine):
    def evaluate_model(self, ckpt: Path) -> dict:
        model = _model_class(self.arch).from_pretrained(ckpt)
        targs = Seq2SeqTrainingArguments(
            output_dir=str(ckpt.parent / "_full_eval_tmp"),
            per_device_eval_batch_size=self.batch_size,
            predict_with_generate=True,
            generation_max_length=self.max_length,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=model, args=targs, processing_class=self.tok,
            data_collator=DataCollatorForSeq2Seq(self.tok, model=model),
            compute_metrics=self.compute_metrics,
        )
        out = {}
        for split, ds in self.tokenized.items():
            out[split] = trainer.evaluate(eval_dataset=ds, metric_key_prefix=f"{split}_full")
        return out


class CausalEngine(_BaseEngine):
    def __init__(self, *a, gen_trainer_cls, collator, sot_id, eos_id, pad_id, **kw):
        super().__init__(*a, **kw)
        self.gen_trainer_cls = gen_trainer_cls
        self.collator = collator
        self.sot_id, self.eos_id, self.pad_id = sot_id, eos_id, pad_id

    def evaluate_model(self, ckpt: Path) -> dict:
        model = _model_class(self.arch).from_pretrained(ckpt)
        targs = TrainingArguments(
            output_dir=str(ckpt.parent / "_full_eval_tmp"),
            per_device_eval_batch_size=self.batch_size,
            report_to="none",
            remove_unused_columns=False,    # we hand-build input_ids/labels
        )
        trainer = self.gen_trainer_cls(
            model=model, args=targs, processing_class=self.tok,
            data_collator=self.collator, compute_metrics=self.compute_metrics,
            sot_id=self.sot_id, eos_id=self.eos_id, pad_id=self.pad_id,
            generation_max_length=self.max_length,
        )
        out = {}
        for split, ds in self.tokenized.items():
            out[split] = trainer.evaluate(eval_dataset=ds, metric_key_prefix=f"{split}_full")
        return out


def build_engine(arch, args, splits) -> _BaseEngine:
    """Load the tokenizer and tokenize each requested split ONCE for this arch."""
    max_length = ARCH_MAX_LENGTH[arch]
    raw = load_dataset(args.dataset)

    def maybe_cap(ds):
        if args.max_eval_samples is None:
            return ds
        n = min(args.max_eval_samples, len(ds))
        return ds.shuffle(seed=0, keep_in_memory=True).select(range(n), keep_in_memory=True)

    if arch == "gpt2_rope":
        gpt2 = _load_runner("gpt2_rope_run")
        tok = AutoTokenizer.from_pretrained(args.tokenizer_gpt2)
        sot_id = tok.convert_tokens_to_ids(gpt2.SOT)
        bos_id, eos_id, pad_id = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
        tokenize = gpt2.make_tokenize_fn(tok, args.src, args.tgt, sot_id, bos_id, eos_id, max_length)
        tokenized = {sp: maybe_cap(raw[sp]).map(tokenize, batched=True,
                                                remove_columns=raw[sp].column_names)
                     for sp in splits}
        return CausalEngine(arch, tok, tokenized, max_length, args.batch_size,
                            gen_trainer_cls=gpt2.GenEvalTrainer,
                            collator=gpt2.CausalLMCollator(pad_token_id=pad_id),
                            sot_id=sot_id, eos_id=eos_id, pad_id=pad_id)
    else:
        vas = _load_runner("vaswani_run")          # seq2seq tokenize is identical for both
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenize = vas.make_tokenize_fn(tok, args.src, args.tgt, max_length)
        tokenized = {sp: maybe_cap(raw[sp]).map(tokenize, batched=True,
                                                remove_columns=raw[sp].column_names)
                     for sp in splits}
        return Seq2SeqEngine(arch, tok, tokenized, max_length, args.batch_size)


def evaluate_one(engine, arch, layers, heads, seed, args) -> dict | None:
    """Evaluate one run's best/ checkpoint; write + return its result dict."""
    ckpt = best_dir(arch, layers, heads, args.epochs, args.weight_decay, seed, args.results_dir)
    rec = {"arch": arch, "layers": layers, "heads": heads, "seed": seed,
           "epochs": args.epochs, "weight_decay": args.weight_decay,
           "ckpt": str(ckpt), "n_eval": args.max_eval_samples}
    if not ckpt.exists():
        print(f"[skip] missing checkpoint: {ckpt}", flush=True)
        rec["status"] = "missing"
        return rec
    metrics = engine.evaluate_model(ckpt)
    rec["status"] = "ok"
    rec["metrics"] = metrics
    rec["exact_match"] = {sp: m.get(f"{sp}_full_exact_match") for sp, m in metrics.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_json_path(arch, layers, heads, args.epochs, args.weight_decay, seed, args.out_dir)
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"[done] {out_path.name}: {rec['exact_match']}", flush=True)
    return rec


def summarize(out_dir: Path) -> None:
    """Scan per-run JSONs and write summary.csv. No eval -- run after an array job."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "summary.csv":
            continue
        r = json.loads(p.read_text())
        if r.get("status") != "ok":
            continue
        em = r.get("exact_match", {})
        for split, val in em.items():
            rows.append((r["arch"], r["layers"], r["heads"], r["seed"], split, val))
    rows.sort()
    header = "arch,layers,heads,seed,split,exact_match"
    lines = [header] + [f"{a},{l},{h},{s},{sp},{v}" for a, l, h, s, sp, v in rows]
    csv_path = out_dir / "summary.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")
    for line in lines:
        print(" ", line)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=[*ARCHS, "all"], default="all")
    p.add_argument("--config", choices=[*CONFIG_KEYS, "all"], default="all",
                   help="Capacity config 'LxH' (layers x heads) to evaluate, or 'all'.")
    p.add_argument("--seeds", default="42-46", help="Inclusive 'A-B' or comma list (default 42-46).")
    p.add_argument("--epochs", type=int, default=1, help="Locates the results dir suffix (match run.py).")
    p.add_argument("--weight-decay", type=float, default=1.0, help="Locates the results dir suffix (match run.py).")
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"),
                   help="137-token v2 seq2seq tokenizer for vaswani / vaswani_rope.")
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"),
                   help="138-token v2 tokenizer (with <sot>) for gpt2_rope.")
    p.add_argument("--src", default="hi")
    p.add_argument("--tgt", default="hf")
    p.add_argument("--splits", default="validation",
                   help="Comma list of splits to score full (default 'validation'; e.g. 'validation,test').")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="Cap each split to this many rows (shuffle seed 0). Default None = FULL split "
                        "-- the whole point of this script. Set a small value only for a sanity check.")
    p.add_argument("--batch-size", type=int, default=128, help="Per-device eval batch size.")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Where the runs' best/ checkpoints live (default this experiment's results/).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Where to write per-run JSONs + summary.csv (default results/full_eval/).")
    p.add_argument("--summarize", action="store_true",
                   help="Don't evaluate; just scan --out-dir JSONs and (re)write summary.csv.")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out_dir)
        return

    seeds = parse_seeds(args.seeds)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # --- Array mode: SLURM_ARRAY_TASK_ID selects one (arch, config, seed) -------
    # Layout IDENTICAL to run.py: task = arch_i*(n_cfg*n_seeds) + cfg_i*n_seeds + seed_i.
    task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_env is not None:
        task = int(task_env)
        n_seeds, n_cfg = len(seeds), len(CONFIGS)
        total = len(ARCHS) * n_cfg * n_seeds
        if task >= total:
            sys.exit(f"SLURM_ARRAY_TASK_ID={task} out of range (use --array=0-{total - 1}).")
        arch = ARCHS[task // (n_cfg * n_seeds)]
        rem = task % (n_cfg * n_seeds)
        layers, heads = CONFIGS[rem // n_seeds]
        seed = seeds[rem % n_seeds]
        print(f"[array] task {task} -> arch={arch} config=L{layers}H{heads} seed={seed} "
              f"splits={splits}", flush=True)
        engine = build_engine(arch, args, splits)
        evaluate_one(engine, arch, layers, heads, seed, args)
        return

    # --- In-process mode: build each arch's engine ONCE, reuse across its runs ---
    archs = ARCHS if args.arch == "all" else [args.arch]
    configs = CONFIGS if args.config == "all" else [CONFIG_BY_KEY[args.config]]
    print(f"[full-eval] archs={archs} configs={[f'L{l}H{h}' for l, h in configs]} "
          f"seeds={seeds} splits={splits} epochs={args.epochs} wd={args.weight_decay} "
          f"max_eval_samples={args.max_eval_samples}", flush=True)
    results = []
    for arch in archs:
        engine = build_engine(arch, args, splits)        # tokenizes full split(s) once
        for (layers, heads) in configs:
            for seed in seeds:
                rec = evaluate_one(engine, arch, layers, heads, seed, args)
                if rec is not None:
                    results.append(rec)

    print("\n=== summary (exact_match) ===", flush=True)
    for r in results:
        if r.get("status") == "ok":
            print(f"  {r['arch']:12s} L{r['layers']}H{r['heads']} seed {r['seed']}: {r['exact_match']}")
        else:
            print(f"  {r['arch']:12s} L{r['layers']}H{r['heads']} seed {r['seed']}: {r['status']}")
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
