"""Train the Vaswani encoder-decoder *with RoPE* on the frame-based HI -> HF dataset.

WHAT'S DIFFERENT FROM EXPERIMENT 06
-----------------------------------
* Architecture: `architecture.modeling_vaswani_rope.VaswaniRoPEForConditionalGeneration`
  — same encoder/decoder scaffolding as `modeling_vaswani.py`, but with the
  fixed sinusoidal positional table replaced by RoPE rotation applied to Q/K
  *inside self-attention* on both encoder and decoder. Cross-attention is left
  un-rotated (no shared position frame between encoder and decoder sequences).
* Everything else (data sources, three-split policy, optimization, eval metric,
  Hub push) is identical to experiment 06, so the two experiments are directly
  comparable as an ablation on positional encoding.

WANDB
-----
project=`constituent-geometry`,
default run=`07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3`.
Set `--no-wandb` to disable; `--run-name` to override.

HF HUB PUSH
-----------
After training, the *best* model (selected on val exact-match) and the
tokenizer are pushed to a PUBLIC Hub repo. The default repo id is
`kylelovesllms/<run-name>`, so swapping `--run-name` (or `--dataset`, which
the run name encodes) also swaps the destination repo. Override with
`--hub-repo-id`; disable with `--no-push`.

Requires authentication before submitting the job:
    huggingface-cli login                   # interactive
    # or: export HF_TOKEN=hf_xxx ...        # in the sbatch script

LABELS / DECODER_START
----------------------
The tokenizer adds <bos>; the model re-creates the decoder input by
`shift_right`-ing the labels and prepending `decoder_start_token_id=<bos>`,
so we strip the leading <bos> from labels to avoid double-counting it.

RUN
---
Local smoke test:
    uv run python experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \\
        --epochs 1 --no-wandb --no-push

Cluster (full run, single GPU):
    sbatch slurm/run_gpu.sbatch \\
        experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from architecture.configuration_vaswani import VaswaniConfig
from architecture.modeling_vaswani_rope import VaswaniRoPEForConditionalGeneration


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
    p.add_argument(
        "--dataset",
        default="kylelovesllms/hi_hf_frames_d3_random_100",
        help="HF Hub id of the frame dataset built by experiment 05 "
             "(must have train/validation/test splits)",
    )
    p.add_argument(
        "--tokenizer",
        default="experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/results",
    )
    p.add_argument("--src", default="hi", help="encoder input column")
    p.add_argument("--tgt", default="hf", help="decoder target column")

    # --- model knobs --------------------------------------------------------
    # Match experiment 06 exactly so the only thing changing across the two
    # runs is the positional-encoding scheme (sinusoidal vs RoPE).
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)

    p.add_argument("--wandb-project", default="constituent-geometry")
    p.add_argument(
        "--run-name",
        default="07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3",
    )
    p.add_argument("--no-wandb", action="store_true")

    # --- HF Hub push --------------------------------------------------------
    p.add_argument(
        "--hub-repo-id",
        default=None,
        help="HF Hub repo id to push best model + tokenizer to. "
             "Default = kylelovesllms/<run-name> (public).",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="disable pushing best model + tokenizer to HF Hub",
    )
    args = p.parse_args()

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = args.run_name

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = load_dataset(args.dataset)
    print(raw)

    train_ds = raw["train"]
    val_ds = raw["validation"]
    test_ds = raw["test"]
    print(
        f"train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}"
    )

    tokenize = make_tokenize_fn(tok, args.src, args.tgt, args.max_length)
    cols = train_ds.column_names
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=cols)
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=cols)

    model = VaswaniRoPEForConditionalGeneration(
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
    print(
        f"model: d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads} d_ff={args.d_ff} -> {n_params:,} params"
    )

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
        # Select on val exact_match (higher-is-better). See exp-04 docstring
        # for why pairing eval_loss with greater_is_better=True is a footgun.
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
        eval_dataset=val_tok,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # --- headline: held-out-frames test exact-match -------------------------
    test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
    print("test (held-out frames):", test_metrics)

    trainer.save_model(f"{args.output_dir}/best")

    # --- push best model + tokenizer to public HF Hub repo ------------------
    # load_best_model_at_end=True means trainer.model IS the best checkpoint
    # (by val exact_match) at this point, so the push captures the same
    # weights that just produced the test_metrics above.
    if not args.no_push:
        repo_id = args.hub_repo_id or f"kylelovesllms/{args.run_name}"
        print(f"pushing best model + tokenizer to https://huggingface.co/{repo_id} (public)")
        trainer.model.push_to_hub(repo_id, private=False)
        tok.push_to_hub(repo_id, private=False)


if __name__ == "__main__":
    main()
