"""Depth-holdout training of the Vaswani encoder-decoder on HI -> HF.

WHY THIS EXPERIMENT EXISTS
--------------------------
Experiment 02 trained on a *random* 80/10/10 split, so train and val shared the
same sentence structures and the same vocabulary. An eval loss of ~0.003 there
tells us the model interpolates well *within seen structure* -- it CANNOT tell us
whether the model learned a recursive reordering algorithm or a shallow,
structure-bound pattern matcher. That distinction is exactly what we need before
any induction-head-style attention analysis is meaningful.

This experiment turns the eval into a DISTRIBUTION SHIFT along recursion depth:

    train + in-distribution val :  sentences with CP-nesting depth in {0, 1}
    OOD generalization test     :  sentences with depth == 2 (never seen)

Depth 0 has no embedding; depth 1 shows the recursive (CP) step exactly once.
If the model then reorders depth-2 sentences correctly, it must have learned to
*apply the rule recursively* rather than memorize depth-<=1 shapes. That is the
positive result that justifies looking for a reorder circuit.

We also SHRINK the model (n_layers=2, n_heads=1 by default) for two reasons:
  1. fewer heads/layers => fewer circuit components to reverse-engineer later;
  2. less capacity nudges the model toward the algorithm rather than lookup.
Find the *smallest* model that still passes the depth-2 test -- that is the ideal
object for interpretability.

NOTE on the data prep: src/training_loops/seq2seq.py (and its prepare_splits) is
not present in the repo, so this script is self-contained. The only subtlety it
reproduces is that labels drop their leading <bos>: the tokenizer adds <bos>, but
the seq2seq model re-creates the decoder input by shift_right-ing the labels and
prepending decoder_start_token_id (=<bos>) itself, so a <bos> left in the labels
would be double-counted.

Run (local):
    uv run python experiments/03_holdout_depth_2_vaswani_hi_hf/run.py --epochs 30
Run (cluster):
    sbatch slurm/run_gpu.sbatch experiments/03_holdout_depth_2_vaswani_hi_hf/run.py --epochs 30
"""

import argparse
import os

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from architecture.configuration_vaswani import VaswaniConfig
from architecture.modeling_vaswani import VaswaniForConditionalGeneration


def parse_depths(spec: str) -> list[int]:
    """'0,1' -> [0, 1]. Used for the --train-depths flag."""
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def pool_by_depth(dsdict, depths: list[int]):
    """Re-partition the dataset by `depth`, ignoring the original random splits.

    Experiment 01's split mixed depths randomly across train/val/test, so to hold
    out a depth cleanly we concatenate everything back together first, then filter.
    """
    full = concatenate_datasets([dsdict[s] for s in dsdict.keys()])
    return full.filter(lambda row: row["depth"] in depths)


def make_tokenize_fn(tok, src: str, tgt: str, max_length: int):
    """Build the map() function: encode `src` as inputs, `tgt` as labels.

    Labels drop the leading <bos> (see module docstring for why).
    """
    def tokenize(batch):
        model_inputs = tok(batch[src], max_length=max_length, truncation=True)
        labels = tok(batch[tgt], max_length=max_length, truncation=True)
        model_inputs["labels"] = [ids[1:] for ids in labels["input_ids"]]
        return model_inputs

    return tokenize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/hi_hf_dataset_depth_3")
    p.add_argument("--tokenizer", default="data/tokenizer")
    p.add_argument(
        "--output-dir",
        default="experiments/03_holdout_depth_2_vaswani_hi_hf/results",
    )
    p.add_argument("--src", default="hi", help="encoder input column")
    p.add_argument("--tgt", default="hf", help="decoder target column")

    # --- the depth-holdout knobs --------------------------------------------
    p.add_argument("--train-depths", default="0,1",
                   help="depths used for train + in-distribution val")
    p.add_argument("--eval-depth", type=int, default=2,
                   help="held-out depth used ONLY for the final OOD test")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of the train-depths pool held out as in-dist val")
    p.add_argument("--max-train", type=int, default=None,
                   help="optional cap on train rows (to probe sample efficiency)")
    p.add_argument("--max-eval", type=int, default=None,
                   help="optional cap on the OOD eval set. Autoregressive generation "
                        "has no KV cache, so the full ~98k depth-2 set takes ~1h on MPS. "
                        "A random sample of e.g. 2000 gives a stable exact-match estimate "
                        "in ~1 min. Leave unset (full set) for the cluster.")


    # --- shrunk, interp-friendly model defaults -----------------------------
    # n_heads=1 trivially satisfies d_model % n_heads == 0 for any d_model.
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=1)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    # The depth-{0,1} pool is tiny (~6.5k rows), so it converges in few steps;
    # eval often is cheap and informative here.
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)

    p.add_argument("--wandb-project", default="allegro")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.run_name:
            os.environ["WANDB_NAME"] = args.run_name

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = load_from_disk(args.dataset)

    # --- build the depth-based splits ---------------------------------------
    train_depths = parse_depths(args.train_depths)
    pool = pool_by_depth(raw, train_depths)              # depth in {0,1}
    ood = pool_by_depth(raw, [args.eval_depth])          # depth == 2
    
    if args.max_eval is not None and len(ood) > args.max_eval:
        ood = ood.shuffle(seed=args.seed).select(range(args.max_eval))


    split = pool.train_test_split(test_size=args.val_frac, seed=args.seed)
    train_ds, val_ds = split["train"], split["test"]
    if args.max_train is not None:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_train))

    print(f"train depths {train_depths}: {len(train_ds):,} rows | "
          f"in-dist val: {len(val_ds):,} | OOD depth-{args.eval_depth}: {len(ood):,}")

    tokenize = make_tokenize_fn(tok, args.src, args.tgt, args.max_length)
    cols = train_ds.column_names
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=cols)
    ood_tok = ood.map(tokenize, batched=True, remove_columns=cols)

    model = VaswaniForConditionalGeneration(
        VaswaniConfig(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    )
    n_params = sum(t.numel() for t in model.parameters())
    print(f"model: d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads} -> {n_params:,} params")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_txt = tok.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tok.batch_decode(labels, skip_special_tokens=True)
        return {"exact_match": float(np.mean(
            [p.strip() == r.strip() for p, r in zip(pred_txt, ref_txt)]
        ))}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=200,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        # Select on the IN-DISTRIBUTION val set. exact_match is higher-is-better,
        # so greater_is_better=True is correct here. (Experiment 02 paired
        # eval_loss with greater_is_better=True, which selects the WORST
        # checkpoint -- fixed by switching the metric to exact_match.)
        metric_for_best_model="exact_match",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        logging_steps=50,
        seed=args.seed,
        report_to=("none" if args.no_wandb else "wandb"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,            # in-distribution val ONLY for selection
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # --- the headline number: OOD generalization, never tuned on ------------
    ood_metrics = trainer.evaluate(
        eval_dataset=ood_tok, metric_key_prefix=f"ood_depth{args.eval_depth}"
    )
    print("OOD generalization:", ood_metrics)

    trainer.save_model(f"{args.output_dir}/best")


if __name__ == "__main__":
    main()
