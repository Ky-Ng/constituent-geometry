"""Full-validation eval suite for experiment 18.

Loads a saved checkpoint per (arch, seed) and reports exact-match on the ENTIRE
validation split (no per-step cap). Built to evaluate the ~epoch-1.0 checkpoints
(default checkpoint-34000) of the converged 5-epoch runs.

It REUSES each runner's arch-specific machinery so the eval matches training-time
eval exactly:
  - seq2seq (vaswani / vaswani_rope): Seq2SeqTrainer + predict_with_generate,
    tokenization via vaswani_run.make_tokenize_fn.
  - gpt2_rope: the custom GenEvalTrainer + CausalLMCollator + make_tokenize_fn
    imported from gpt2_rope_run.
Models are loaded with from_pretrained() from
  results/<arch>/<suffix>/seed_<S>/checkpoint-<step>/.
Importing the runner modules only runs their module-level defs (their main() is
__main__-guarded), so nothing trains.

Usage (one arch, all 5 seeds, full validation):
    uv run python experiments/18_multi_seed_random_depth2_grok/eval_full_val.py \\
        --arch vaswani --seeds 42-46 --checkpoint-step 34000

Quick smoke (small subset, no GPU needed):
    ... --arch vaswani --seeds 42 --max-eval-samples 64
"""

import argparse
import json
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
sys.path.insert(0, str(EXP_DIR / "runners"))

from architecture.modeling_vaswani import VaswaniForConditionalGeneration
from architecture.modeling_vaswani_rope import VaswaniRoPEForConditionalGeneration
from architecture.modeling_gpt2_rope import GPT2RoPEForCausalLM
import vaswani_run as _seq2seq          # make_tokenize_fn(tok, src, tgt, max_length)
import gpt2_rope_run as _gpt2           # make_tokenize_fn(...), GenEvalTrainer, CausalLMCollator, SOT

SEQ2SEQ_MODEL = {
    "vaswani": VaswaniForConditionalGeneration,
    "vaswani_rope": VaswaniRoPEForConditionalGeneration,
}
# default generation length per arch (matches the runners' --max-length defaults:
# seq2seq target stream is short; gpt2 sees the concatenated hi<sot>hf stream).
DEFAULT_MAX_LEN = {"vaswani": 64, "vaswani_rope": 64, "gpt2_rope": 128}


def parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def make_compute_metrics(tok):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_txt = tok.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tok.batch_decode(labels, skip_special_tokens=True)
        return {
            "exact_match": float(
                np.mean([p.strip() == r.strip() for p, r in zip(pred_txt, ref_txt)])
            )
        }
    return compute_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["vaswani", "vaswani_rope", "gpt2_rope"])
    p.add_argument("--seeds", default="42-46")
    p.add_argument("--results-dir", default=str(EXP_DIR / "results"))
    p.add_argument("--suffix", default="ep5_wd1.0", help="results/<arch>/<suffix>/seed_<S>/")
    p.add_argument("--checkpoint-step", type=int, default=34000,
                   help="which checkpoint-<step> to load (default 34000 ~= epoch 0.98).")
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random")
    p.add_argument("--split", default="validation")
    p.add_argument("--tokenizer", default=str(EXP_DIR / "artifacts" / "tokenizer"))
    p.add_argument("--tokenizer-gpt2", default=str(EXP_DIR / "artifacts" / "tokenizer_gpt2"))
    p.add_argument("--src", default="hi")
    p.add_argument("--tgt", default="hf")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=None, help="default: 64 seq2seq / 128 gpt2")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="cap eval rows (smoke only); default None = ENTIRE split.")
    p.add_argument("--out", default=None, help="optional JSON path for the results")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    max_length = args.max_length or DEFAULT_MAX_LEN[args.arch]
    tok_path = args.tokenizer_gpt2 if args.arch == "gpt2_rope" else args.tokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)

    raw = load_dataset(args.dataset)
    val = raw[args.split]
    if args.max_eval_samples is not None:
        n = min(args.max_eval_samples, len(val))
        val = val.shuffle(seed=0, keep_in_memory=True).select(range(n), keep_in_memory=True)
    print(f"[{args.arch}] eval split='{args.split}' rows={len(val):,} "
          f"checkpoint-step={args.checkpoint_step} max_length={max_length}", flush=True)

    # arch-specific tokenization (done once; reused across seeds)
    if args.arch == "gpt2_rope":
        sot_id = tok.convert_tokens_to_ids(_gpt2.SOT)
        bos_id, eos_id, pad_id = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
        tokenize = _gpt2.make_tokenize_fn(tok, args.src, args.tgt, sot_id, bos_id, eos_id, max_length)
    else:
        tokenize = _seq2seq.make_tokenize_fn(tok, args.src, args.tgt, max_length)
    val_tok = val.map(tokenize, batched=True, remove_columns=val.column_names)

    compute_metrics = make_compute_metrics(tok)
    tmp_out = f"/tmp/eval18_{args.arch}"
    results: dict[str, float] = {}

    for seed in seeds:
        ckpt = Path(args.results_dir) / args.arch / args.suffix / f"seed_{seed}" / f"checkpoint-{args.checkpoint_step}"
        if not ckpt.exists():
            print(f"[{args.arch} seed {seed}] MISSING {ckpt} -- skipping", flush=True)
            continue

        if args.arch == "gpt2_rope":
            model = GPT2RoPEForCausalLM.from_pretrained(str(ckpt))
            targs = TrainingArguments(
                output_dir=tmp_out, per_device_eval_batch_size=2 * args.batch_size,
                report_to="none", remove_unused_columns=False,
            )
            trainer = _gpt2.GenEvalTrainer(
                model=model, args=targs, eval_dataset=val_tok, processing_class=tok,
                data_collator=_gpt2.CausalLMCollator(pad_token_id=pad_id),
                compute_metrics=compute_metrics,
                sot_id=sot_id, eos_id=eos_id, pad_id=pad_id, generation_max_length=max_length,
            )
        else:
            model = SEQ2SEQ_MODEL[args.arch].from_pretrained(str(ckpt))
            targs = Seq2SeqTrainingArguments(
                output_dir=tmp_out, per_device_eval_batch_size=2 * args.batch_size,
                predict_with_generate=True, generation_max_length=max_length, report_to="none",
            )
            trainer = Seq2SeqTrainer(
                model=model, args=targs, eval_dataset=val_tok, processing_class=tok,
                data_collator=DataCollatorForSeq2Seq(tok, model=model),
                compute_metrics=compute_metrics,
            )

        metrics = trainer.evaluate(metric_key_prefix="val_full")
        em = metrics.get("val_full_exact_match")
        results[f"seed_{seed}"] = em
        print(f"[{args.arch} seed {seed}] full-{args.split} exact_match = {em:.4f} "
              f"(n={len(val_tok):,}, runtime={metrics.get('val_full_runtime')}s)", flush=True)

    # summary
    vals = [v for v in results.values() if v is not None]
    if vals:
        print(f"\n=== {args.arch} full-{args.split} exact-match over {len(vals)} seed(s) ===", flush=True)
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")
        print(f"  mean={np.mean(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}", flush=True)

    out_path = args.out or str(EXP_DIR / "artifacts" / f"full_{args.split}_eval_{args.arch}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "arch": args.arch, "split": args.split, "checkpoint_step": args.checkpoint_step,
            "n_rows": len(val_tok), "dataset": args.dataset, "results": results,
        }, f, indent=2)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
