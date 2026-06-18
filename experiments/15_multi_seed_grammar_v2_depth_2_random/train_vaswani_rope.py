"""Train the Vaswani+RoPE encoder-decoder for experiment 15 (v2 grammar + adjunction).

Copy of experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py with
two additions:
* Default tokenizer/dataset point to the v2-adjunction artifacts.
* --stop-threshold: when set, training stops 1 full epoch after
  eval_exact_match >= threshold (via ThresholdStopCallback).

All architecture hyperparameters, training args, and Hub push logic are
identical to experiment 07.

To copy: rename this file to train_vaswani_rope.py.
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
from training_loops.callbacks import ThresholdStopCallback

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def make_tokenize_fn(tok, src: str, tgt: str, max_length: int):
    def tokenize(batch):
        model_inputs = tok(batch[src], max_length=max_length, truncation=True)
        labels = tok(batch[tgt], max_length=max_length, truncation=True)
        model_inputs["labels"] = [ids[1:] for ids in labels["input_ids"]]
        return model_inputs
    return tokenize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-adj-frames-d2-k100-random")
    p.add_argument("--tokenizer",
                   default=os.path.join(EXP_DIR, "artifacts", "tokenizer"))
    p.add_argument("--output-dir",
                   default=os.path.join(EXP_DIR, "results", "vaswani_rope", "seed_0"))
    p.add_argument("--src", default="hi")
    p.add_argument("--tgt", default="hf")
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
    p.add_argument("--run-name",
                   default="15_multi_seed_grammar_v2_depth_2_random_vaswani_rope_seed_0")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--hub-repo-id", default=None)
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--stop-threshold", type=float, default=None,
        help="Stop 1 epoch after eval_exact_match >= this value. "
             "Default None = train for all --epochs.",
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
    print(f"train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")

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
    print(f"model: d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads} d_ff={args.d_ff} -> {n_params:,} params")

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
        metric_for_best_model="exact_match",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        logging_steps=50,
        seed=args.seed,
        report_to=("none" if args.no_wandb else "wandb"),
    )

    callbacks = []
    if args.stop_threshold is not None:
        callbacks.append(ThresholdStopCallback(threshold=args.stop_threshold))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
    )
    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
    print("test (held-out frames):", test_metrics)

    trainer.save_model(f"{args.output_dir}/best")

    if not args.no_push:
        repo_id = args.hub_repo_id or f"kylelovesllms/{args.run_name}"
        print(f"pushing best model + tokenizer to https://huggingface.co/{repo_id} (public)")
        trainer.model.push_to_hub(repo_id, private=False)
        tok.push_to_hub(repo_id, private=False)


if __name__ == "__main__":
    main()
