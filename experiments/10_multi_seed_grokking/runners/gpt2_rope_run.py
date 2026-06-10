"""Train a GPT2-style decoder-only Transformer with RoPE on the HI -> HF dataset.

EXPERIMENT-10 COPY (grokking sweep)
-----------------------------------
This is a *verbatim copy* of
`experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py` with TWO added
knobs threaded into TrainingArguments: `--weight-decay` and `--lr-scheduler`. We
copy rather than edit 08 because repo convention forbids editing existing
experiments, and grokking needs weight decay and a non-decaying LR (08 hardcodes
0.0 / 'linear'). The default `--tokenizer`
still points at 08's `<sot>`-augmented artifact dir, which stays in the repo, so
the copy resolves it identically when run with cwd=REPO_ROOT. Everything else
(GenEvalTrainer, CausalLMCollator, loss masking, Hub push) is unchanged. Driven
by `experiments/10_multi_seed_grokking/run.py`.

DESIGN
------
* **Decoder-only causal LM** over the concatenated sequence
  ``input_ids = [<bos>] + hi + [<sot>] + hf + [<eos>]``.
* **Loss masking**: labels are a copy of ``input_ids`` with positions
  ``0 .. sot_pos`` (inclusive) set to ``-100``. The model's CE loss shifts
  internally (position t predicts ``labels[t+1]``), so the first un-masked
  prediction uses input ``<sot>`` at position ``sot_pos`` to predict ``hf_1``
  at position ``sot_pos + 1``. The final ``<eos>`` is unmasked so the model
  learns when to stop generating.
* **Model**: ``architecture.modeling_gpt2_rope.GPT2RoPEForCausalLM`` -- GPT2
  block (pre-LN, GELU, biases, final LN, tied LM head) with RoPE on Q/K and no
  learned/sinusoidal positional embedding.
* **Eval**: a custom Trainer subclass (`GenEvalTrainer`) carves the prompt
  ``[<bos>] + hi + [<sot>]`` out of each eval row, runs greedy
  ``model.generate``, strips the prompt, and returns the continuation as
  ``predictions`` so the standard ``compute_metrics`` interface still works.
  This matches the ``metric_for_best_model="exact_match"`` selection rule used
  by experiments 06 and 07.

WANDB
-----
project=`constituent-geometry`,
default run=`08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3`.

HF HUB PUSH
-----------
After training, the *best* model (selected on val ``exact_match``) and the
tokenizer are pushed to a PUBLIC Hub repo. Default repo id =
``kylelovesllms/<run-name>``. Override with ``--hub-repo-id``; disable with
``--no-push``.

Requires authentication:
    huggingface-cli login            # interactive
    # or: export HF_TOKEN=hf_xxx     # in the sbatch script

PREREQ
------
Uses the SOT-augmented tokenizer built once by experiment 08:
    uv run python experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/build_tokenizer.py

RUN
---
Local smoke test:
    uv run python experiments/10_multi_seed_grokking/runners/gpt2_rope_run.py \\
        --epochs 1 --no-wandb --no-push

Driven by the experiment-10 launcher (`experiments/10_multi_seed_grokking/run.py`).
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


SOT = "<sot>"


# =============================================================================
# Tokenize and label-mask: build the concatenated sequence + the masked labels.
# =============================================================================
def make_tokenize_fn(tok, src: str, tgt: str, sot_id: int, bos_id: int,
                     eos_id: int, max_length: int):
    """Return a ``map(batched=True)`` function that, for each row:

      input_ids      = [<bos>] + hi_ids + [<sot>] + hf_ids + [<eos>]
      labels         = -100 for every position up to and INCLUDING <sot>,
                       then hf_1, ..., hf_m, <eos> verbatim.
      attention_mask = 1 everywhere (padding is added later by the collator).

    Truncation policy: if the joined sequence exceeds ``max_length`` we
    truncate from the RIGHT, which can drop <eos> or trail of ``hf``. That is
    an acceptable loss-of-supervision case for now; bump ``max_length`` if it
    bites (the long tail of depth-4 sentences is the likely culprit, same as
    in experiments 06/07).
    """
    def tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for hi_text, hf_text in zip(batch[src], batch[tgt]):
            hi_ids = tok(hi_text, add_special_tokens=False)["input_ids"]
            hf_ids = tok(hf_text, add_special_tokens=False)["input_ids"]
            ids = [bos_id] + hi_ids + [sot_id] + hf_ids + [eos_id]
            ids = ids[:max_length]
            sot_pos = 1 + len(hi_ids)  # index of <sot> inside `ids`
            # Mask 0..sot_pos inclusive (so positions whose PREDICTED token is
            # part of {<bos>, hi, <sot>} contribute no loss after the internal
            # shift). If truncation cut the sequence to below sot_pos, mask
            # everything -- it's a degenerate row.
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
    """Right-pad ``input_ids`` / ``attention_mask`` / ``labels`` to the longest
    sequence in the batch. We can't reuse ``DataCollatorForLanguageModeling``
    because it would overwrite our carefully masked labels with a fresh copy of
    ``input_ids``."""

    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * n_pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * n_pad)
            # -100 on pad positions: the loss never sees them.
            batch["labels"].append(f["labels"] + [-100] * n_pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


# =============================================================================
# Generation-based eval Trainer.
# =============================================================================
class GenEvalTrainer(Trainer):
    """Trainer that generates from ``[<bos>] + hi + [<sot>]`` during eval and
    returns the generated continuation as ``predictions``, with the gold hf
    ids as ``label_ids``.

    Why subclass at all?
      ``Seq2SeqTrainer`` assumes an encoder-decoder model and feeds
      ``decoder_input_ids`` derived from ``labels``. A causal LM doesn't have
      that path: the prompt has to be sliced out of the *input* sequence
      (everything up to and including <sot>), which Seq2SeqTrainer can't do.

    Padding policy:
      All returned ``preds`` and ``labels`` tensors are padded to a fixed
      length ``generation_max_length`` (with the pad token id). That ensures
      Trainer's cross-batch nested-concat sees a uniform dim 1 across every
      eval batch and never has to reshape.
    """

    def __init__(self, *args, sot_id: int, eos_id: int, pad_id: int,
                 generation_max_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sot_id = sot_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.generation_max_length = generation_max_length

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 1) Compute eval loss the normal way -- uses the full padded sequence
        #    with our masked labels, so it's an apples-to-apples loss against
        #    training. (Useful for sanity even though we *select* on EM.)
        with torch.no_grad():
            loss_out = model(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                labels=inputs["labels"].to(model.device),
            )
            loss = loss_out.loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        # 2) Build a prompt batch from each row: keep [..., <sot>], drop the
        #    rest. The first non-(-100) label position is sot_pos + 1, i.e. the
        #    first hf token -- so the prompt is row_ids[:first_hf].
        prompts, prompt_attn, golds = [], [], []
        ids_list = inputs["input_ids"].tolist()
        labels_list = inputs["labels"].tolist()
        for row_ids, row_labels in zip(ids_list, labels_list):
            try:
                first_hf = next(i for i, v in enumerate(row_labels) if v != -100)
            except StopIteration:
                # Degenerate row (e.g. fully truncated) -- skip gen, gold is empty.
                first_hf = len(row_ids)
            prompt = row_ids[:first_hf]
            gold = [v for v in row_labels if v != -100]
            prompts.append(prompt)
            prompt_attn.append([1] * len(prompt))
            golds.append(gold)

        # 3) Left-pad prompts (HF causal-LM generation requires left padding so
        #    the last token of every row is the most recent context token).
        max_p = max(len(p) for p in prompts)
        padded_ids, padded_attn = [], []
        for p, a in zip(prompts, prompt_attn):
            n = max_p - len(p)
            padded_ids.append([self.pad_id] * n + p)
            padded_attn.append([0] * n + a)
        input_ids = torch.tensor(padded_ids, device=model.device)
        attention_mask = torch.tensor(padded_attn, device=model.device)

        # 4) Greedy generate. Cap total length at ``generation_max_length`` so
        #    the cross-batch shape stays uniform after right-padding.
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
        # Strip the prompt prefix; what's left is the model's continuation.
        gen_only = gen[:, input_ids.shape[1]:].cpu().tolist()

        # 5) Right-pad both preds and golds to a fixed length so Trainer's
        #    cross-batch concat sees uniform shapes. We use the model's
        #    ``generation_max_length`` as the canonical L: it's always >= the
        #    longest continuation we could have produced.
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


# =============================================================================
# Main.
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="kylelovesllms/hi_hf_frames_d3_random_100",
        help="HF Hub id of the frame dataset built by experiment 05.",
    )
    p.add_argument(
        "--tokenizer",
        default="experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer",
        help="Path to a tokenizer built by build_tokenizer.py in this folder.",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/results",
    )
    p.add_argument("--src", default="hi", help="source column (prompt)")
    p.add_argument("--tgt", default="hf", help="target column (completion)")

    # --- model knobs --------------------------------------------------------
    # Match 06/07 width and shape so per-parameter capacity is comparable. The
    # concatenated sequence is ~2x longer than 06/07's target-only sequence,
    # so we bump max_position_embeddings (and --max-length) to 128.
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

    # --- grokking knobs (the ONLY additions vs experiment 08) ---------------
    # AdamW L2 regularization. 08 leaves this at the HF default (0.0); grokking
    # studies use a nonzero value to pressure the model off the memorizing
    # solution toward the generalizing one. The launcher sweeps it.
    p.add_argument("--weight-decay", type=float, default=0.0)
    # LR schedule. 08 uses the HF default 'linear' (warmup then decay to 0),
    # which leaves LR ~0 by the late epochs where grokking would occur. The
    # experiment-10 launcher overrides this to 'constant_with_warmup' (flat LR
    # after warmup) so late-epoch learning isn't starved and runs extend cleanly.
    p.add_argument("--lr-scheduler", default="linear")

    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)

    p.add_argument("--wandb-project", default="constituent-geometry")
    p.add_argument(
        "--run-name",
        default="08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3",
    )
    p.add_argument("--no-wandb", action="store_true")

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
    sot_id = tok.convert_tokens_to_ids(SOT)
    if sot_id is None or sot_id == tok.unk_token_id:
        raise ValueError(
            f"Tokenizer at {args.tokenizer} has no {SOT} token. "
            f"Build it first with build_tokenizer.py in this experiment folder."
        )
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    raw = load_dataset(args.dataset)
    print(raw)

    train_ds = raw["train"]
    val_ds = raw["validation"]
    test_ds = raw["test"]
    print(
        f"train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}"
    )

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
    print(
        f"model: d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads} d_ff={args.d_ff} -> {n_params:,} params"
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # both [N, generation_max_length]
        # batch_decode with skip_special_tokens=True strips <pad>/<bos>/<eos>/
        # <sot>, leaving only the hf word stream. We compare those strings.
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
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=200,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        # Select on val exact_match (higher-is-better). Same selection rule as
        # 06/07. See exp-04 docstring for why pairing eval_loss with
        # greater_is_better=True is a footgun.
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to=("none" if args.no_wandb else "wandb"),
        # We hand-build input_ids / attention_mask / labels in tokenize(); the
        # Trainer's default ``remove_unused_columns=True`` would strip them.
        remove_unused_columns=False,
    )

    trainer = GenEvalTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tok,
        data_collator=CausalLMCollator(pad_token_id=pad_id),
        compute_metrics=compute_metrics,
        sot_id=sot_id,
        eos_id=eos_id,
        pad_id=pad_id,
        generation_max_length=args.max_length,
    )
    trainer.train()

    # --- headline: held-out-frames test exact-match -------------------------
    test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
    print("test (held-out frames):", test_metrics)

    trainer.save_model(f"{args.output_dir}/best")

    # --- push best model + tokenizer to public HF Hub repo ------------------
    if not args.no_push:
        repo_id = args.hub_repo_id or f"kylelovesllms/{args.run_name}"
        print(
            f"pushing best model + tokenizer to "
            f"https://huggingface.co/{repo_id} (public)"
        )
        trainer.model.push_to_hub(repo_id, private=False)
        tok.push_to_hub(repo_id, private=False)


if __name__ == "__main__":
    main()
