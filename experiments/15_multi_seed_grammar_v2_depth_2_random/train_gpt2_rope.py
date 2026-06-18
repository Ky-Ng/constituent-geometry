"""Train a GPT2-style decoder-only Transformer with RoPE for experiment 15.

Copy of experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py with
two additions:
* Default tokenizer points to artifacts/tokenizer_gpt2 (v2 grammar + <sot>),
  built by build_tokenizer_gpt2.py in this experiment folder.
* --stop-threshold: when set, training stops 1 full epoch after
  eval_exact_match >= threshold (via ThresholdStopCallback).

PREREQ: build the SOT-augmented v2 tokenizer first:
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/build_tokenizer_gpt2.py

All architecture hyperparameters, training args, and Hub push logic are
identical to experiment 08.

To copy: rename this file to train_gpt2_rope.py.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from architecture.configuration_gpt2_rope import GPT2RoPEConfig
from architecture.modeling_gpt2_rope import GPT2RoPEForCausalLM
from training_loops.callbacks import ThresholdStopCallback

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

SOT = "<sot>"


def make_tokenize_fn(tok, src: str, tgt: str, sot_id: int, bos_id: int,
                     eos_id: int, max_length: int):
    def tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for hi_text, hf_text in zip(batch[src], batch[tgt]):
            hi_ids = tok(hi_text, add_special_tokens=False)["input_ids"]
            hf_ids = tok(hf_text, add_special_tokens=False)["input_ids"]
            ids = [bos_id] + hi_ids + [sot_id] + hf_ids + [eos_id]
            ids = ids[:max_length]
            sot_pos = 1 + len(hi_ids)
            if sot_pos >= len(ids):
                labels = [-100] * len(ids)
            else:
                labels = [-100] * (sot_pos + 1) + ids[sot_pos + 1:]
            assert len(labels) == len(ids)
            out["input_ids"].append(ids)
            out["attention_mask"].append([1] * len(ids))
            out["labels"].append(labels)
        return out
    return tokenize


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * n_pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * n_pad)
            batch["labels"].append(f["labels"] + [-100] * n_pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


class GenEvalTrainer(Trainer):
    def __init__(self, *args, sot_id: int, eos_id: int, pad_id: int,
                 generation_max_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sot_id = sot_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.generation_max_length = generation_max_length

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss_out = model(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                labels=inputs["labels"].to(model.device),
            )
            loss = loss_out.loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        prompts, prompt_attn, golds = [], [], []
        ids_list = inputs["input_ids"].tolist()
        labels_list = inputs["labels"].tolist()
        for row_ids, row_labels in zip(ids_list, labels_list):
            try:
                first_hf = next(i for i, v in enumerate(row_labels) if v != -100)
            except StopIteration:
                first_hf = len(row_ids)
            prompt = row_ids[:first_hf]
            gold = [v for v in row_labels if v != -100]
            prompts.append(prompt)
            prompt_attn.append([1] * len(prompt))
            golds.append(gold)

        max_p = max(len(p) for p in prompts)
        padded_ids, padded_attn = [], []
        for p, a in zip(prompts, prompt_attn):
            n = max_p - len(p)
            padded_ids.append([self.pad_id] * n + p)
            padded_attn.append([0] * n + a)
        input_ids = torch.tensor(padded_ids, device=model.device)
        attention_mask = torch.tensor(padded_attn, device=model.device)

        max_new = max(1, self.generation_max_length - input_ids.shape[1])
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_id,
            )
        gen_only = gen[:, input_ids.shape[1]:].cpu().tolist()

        L = self.generation_max_length

        def _pad(seq):
            return seq[:L] + [self.pad_id] * max(0, L - len(seq))

        preds = torch.tensor(
            [_pad(g) for g in gen_only], dtype=torch.long, device=model.device
        )
        gold = torch.tensor(
            [_pad(g) for g in golds], dtype=torch.long, device=model.device
        )
        return (loss, preds, gold)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="kylelovesllms/hi-hf-v2-adj-frames-d2-k100-random")
    p.add_argument("--tokenizer",
                   default=os.path.join(EXP_DIR, "artifacts", "tokenizer_gpt2"))
    p.add_argument("--output-dir",
                   default=os.path.join(EXP_DIR, "results", "gpt2_rope", "seed_0"))
    p.add_argument("--src", default="hi")
    p.add_argument("--tgt", default="hf")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--wandb-project", default="constituent-geometry")
    p.add_argument("--run-name",
                   default="15_multi_seed_grammar_v2_depth_2_random_gpt2_rope_seed_0")
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
    sot_id = tok.convert_tokens_to_ids(SOT)
    if sot_id is None or sot_id == tok.unk_token_id:
        raise ValueError(
            f"Tokenizer at {args.tokenizer} has no {SOT} token. "
            f"Build it first with build_tokenizer_gpt2.py in this experiment folder."
        )
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    raw = load_dataset(args.dataset)
    print(raw)

    train_ds = raw["train"]
    val_ds = raw["validation"]
    test_ds = raw["test"]
    print(f"train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")

    tokenize = make_tokenize_fn(
        tok, args.src, args.tgt, sot_id, bos_id, eos_id, args.max_length
    )
    cols = train_ds.column_names
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=cols)
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=cols)

    config = GPT2RoPEConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_position_embeddings=args.max_length,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        sot_token_id=sot_id,
    )
    model = GPT2RoPEForCausalLM(config)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"model: d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads} d_ff={args.d_ff} -> {n_params:,} params")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_txt = tok.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tok.batch_decode(labels, skip_special_tokens=True)
        return {
            "exact_match": float(
                np.mean([p.strip() == r.strip() for p, r in zip(pred_txt, ref_txt)])
            )
        }

    targs = TrainingArguments(
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
        logging_steps=50,
        seed=args.seed,
        report_to=("none" if args.no_wandb else "wandb"),
        remove_unused_columns=False,
    )

    callbacks = []
    if args.stop_threshold is not None:
        callbacks.append(ThresholdStopCallback(threshold=args.stop_threshold))

    trainer = GenEvalTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tok,
        data_collator=CausalLMCollator(pad_token_id=pad_id),
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
        sot_id=sot_id,
        eos_id=eos_id,
        pad_id=pad_id,
        generation_max_length=args.max_length,
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
