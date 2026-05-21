"""Build a HuggingFace dataset of Head-Initial / Head-Final sentence pairs.

Exhaustively enumerates every distinct sentence with CP-nesting depth
<= --max-depth, collects them into a Dataset, makes an 80/10/10 split, saves to
local disk, and optionally pushes to the Hub.

--n is a safety CAP on the number of rows (default 1,677,696 == all sentences up
to depth 3). The default depth (3) produces exactly that many, so the default
run collects the full depth-3 set and stops. The cap exists so a larger
--max-depth doesn't silently try to build ~27M rows (depth 4) and exhaust RAM.

Generate the default depth-3 set (~1.68M rows, ~3 GB peak RAM, ~1 GB on disk):
    uv run python experiments/01_build_simple_hi_hf_ds/run.py

Smaller set:
    uv run python experiments/01_build_simple_hi_hf_ds/run.py --max-depth 2

Push to the Hub (huggingface_hub ships with `datasets`; run
`uv run huggingface-cli login` once with a WRITE token first):
    uv run python experiments/01_build_simple_hi_hf_ds/run.py --max-depth 2 --push your-username/hi-hf-toy-cfg

Schema (one row per unique sentence pair):
    id            str    zero-padded, stable, assigned in enumeration order
    hi            str    head-initial sentence
    hf            str    head-final sentence
    hi_bracketed  str    head-initial constituency parse
    hf_bracketed  str    head-final constituency parse
    depth         int    number of embedded clauses (CP nesting)
    n_tokens      int    number of terminal tokens (same for hi and hf)
"""

import argparse
from itertools import islice

from datasets import Dataset, DatasetDict
from tqdm import tqdm  # ships with `datasets`, no extra dependency

# src/ is the import root in this repo (cf. experiments/00_example/run.py).
from grammar.generate import count_sentences, enumerate_pairs


def _row(idx: int, pair) -> dict:
    """Turn one SentencePair into a dataset row with id/depth/n_tokens added.

    `depth` is read off the bracketed parse by counting CP nodes. In this grammar
    each VP takes at most one CP complement and each CP embeds exactly one S, so
    the CP nodes form a single chain — the count equals the nesting depth. (If the
    grammar ever allowed sibling CPs, track depth in the generator instead.)
    """
    return {
        "id": f"{idx:06d}",
        "hi": pair.hi,
        "hf": pair.hf,
        "hi_bracketed": pair.hi_bracketed,
        "hf_bracketed": pair.hf_bracketed,
        "depth": pair.hi_bracketed.count("[CP "),
        "n_tokens": len(pair.hi_tokens),
    }


def build(max_depth: int, split_seed: int, cap: int) -> DatasetDict:
    total = count_sentences(max_depth)
    target = min(total, cap)  # how many rows we'll actually collect
    print(f"depth <= {max_depth}: {total:,} sentences exist; collecting {target:,} ...")

    # islice enforces the cap, so tqdm's total is exact (no overshoot past `cap`).
    capped = islice(enumerate_pairs(max_depth), cap)
    rows = [
        _row(i, pair)
        for i, pair in enumerate(tqdm(capped, total=target, desc="enumerating", unit="sent"))
    ]
    if total > cap:
        print(f"capped at {cap:,} rows of {total:,} "
              f"(subset is enumeration-order-biased, not a uniform sample)")

    full = Dataset.from_list(rows)

    # Random 80/10/10 split. NOTE: train and test share the same sentence
    # *structures*; only the exact sentences differ. Swap to a depth-held-out
    # split later if you want to measure structural generalization.
    tmp = full.train_test_split(test_size=0.2, seed=split_seed)
    val_test = tmp["test"].train_test_split(test_size=0.5, seed=split_seed)
    return DatasetDict(
        {
            "train": tmp["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=3, help="max nested CP embeddings")
    parser.add_argument(
        "--n",
        type=int,
        default=1_677_696,
        help="safety cap on rows collected (default = full depth-3 set)",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the train/val/test split")
    parser.add_argument(
        "--out",
        type=str,
        default="data/hi_hf_dataset",
        help="local path to save_to_disk (data/ is gitignored)",
    )
    parser.add_argument(
        "--push",
        type=str,
        default=None,
        help="Hub repo id to push to, e.g. your-username/hi-hf-toy-cfg (requires huggingface-cli login)",
    )
    parser.add_argument("--private", action="store_true", help="push the Hub dataset as private")
    args = parser.parse_args()

    ds = build(args.max_depth, args.split_seed, args.n)
    print(ds)

    # (a) Save locally. Reload later with: datasets.load_from_disk(args.out)
    ds.save_to_disk(args.out)
    print(f"saved to {args.out}")

    # (b) Publish to the Hub when --push is given.
    if args.push:
        ds.push_to_hub(args.push, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
