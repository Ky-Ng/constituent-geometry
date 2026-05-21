"""ALTERNATIVE to run.py: train with HuggingFace's Seq2SeqTrainer instead of the
hand-written loop in training_loops.seq2seq.

Same model, same data prep (we reuse prepare_splits so labels still drop the
leading <bos>). What the Trainer gives you for free vs. the custom loop:
  + eval_loss logged automatically (the val/loss you wanted)
  + checkpointing, load_best_model_at_end, wandb wiring, mixed precision
  + predict_with_generate for generation-based metrics
What you give up:
  - the explicit loop you can read line-by-line / hook for interpretability
  - teacher-forced token_acc (generation eval gives sequence exact-match instead)

To copy: rename to run_trainer.py (keep both; pick per experiment).
Run:  uv run python experiments/02_train_vaswani_simple_hi_hf_ds_depth_3/run_trainer.py --epochs 5
"""

import argparse

import numpy as np
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from architecture.configuration_vaswani import VaswaniConfig
from architecture.modeling_vaswani import VaswaniForConditionalGeneration
from training_loops.seq2seq import prepare_splits  # shared, correct label handling


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/hi_hf_dataset_depth_3")
    p.add_argument("--tokenizer", default="data/tokenizer")
    p.add_argument("--output-dir", default="experiments/02_train_vaswani_simple_hi_hf_ds_depth_3/results_trainer")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=32)

    # Eval/checkpoint cadence. Keep save-steps a MULTIPLE of eval-steps, or
    # load_best_model_at_end errors. Small values => eval_loss appears early and
    # often (this task converges in well under 1000 steps).
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)

    p.add_argument("--wandb-project", default="allegro", help="wandb project (sets WANDB_PROJECT)")
    p.add_argument("--run-name", default=None, help="wandb run name (default: wandb auto-generates)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = load_from_disk(args.dataset)
    tokenized = prepare_splits(raw, tok, "hi", "hf", args.max_length)

    model = VaswaniForConditionalGeneration(VaswaniConfig(vocab_size=tok.vocab_size))

    def compute_metrics(eval_pred):
        # predict_with_generate => predictions are generated token ids.
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_txt = tok.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tok.batch_decode(labels, skip_special_tokens=True)
        return {"exact_match": float(np.mean([p.strip() == r.strip() for p, r in zip(pred_txt, ref_txt)]))}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=500,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=args.max_length,
        logging_steps=50,
        report_to=("none" if args.no_wandb else "wandb"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best")  # load_best_model_at_end already restored best weights


if __name__ == "__main__":
    main()
