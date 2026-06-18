"""Pilot: cross-seed attention-heatmap grids for two hand-picked depth-0 prompts.

Same OUTPUT style as compare_heatmaps.py / gather_heatmaps.py -- per (family,
layer) an S x H grid with rows=seeds, cols=heads, columns head-aligned to a
reference seed, annotated with the order-invariant matched cosine. The only
differences from compare_heatmaps.py:

  - Architecture FIXED to vaswani_rope (the arch whose 10 seeds have finished).
  - Each seed uses its LAST checkpoint (max checkpoint-N), not "best".
    NOTE: this mixes training steps across seeds (most are checkpoint-3200,
    seed_43 is checkpoint-6200).  Fine for an eyeball pilot, NOT apples-to-apples.
  - Two explicit (hi, hf) prompts with caller-supplied labels, instead of a
    dataset row.
  - Axis tick labels carry a 0-based positional ID per HI word (the_0 boy_1
    likes_2 a_3 girl_4).  The SAME word keeps its ID on the HF axis after the
    head-final reorder, so you can trace a constituent across HI/HF axes.  This
    only decorates the displayed ticks -- the tokenized text is untouched.

WHY A FIXED GOLD HF (not generated)
-----------------------------------
Head-aligned cross-seed grids require IDENTICAL token axes / tensor shapes across
seeds.  A generated HF can differ in length per seed and break alignment, so we
teacher-force the same gold (hi, hf) for every seed -- exactly compare_heatmaps'
"WHY A FIXED GOLD PAIR" rationale.  The HF here is the grammar's head-final
reorder (subject-DP, object-DP, verb; each DP keeps its D-N order).

All model-loading / extraction / head-matching / plotting is reused from
visualization.attention_heatmaps (ADAPTERS) and compare_heatmaps -- no logic is
duplicated.

OUTPUT
------
    figures/vaswani_rope/heatmaps/<label>/layer_<l>_<family>.png
    results/vaswani_rope/<label>_heatmap_consistency.csv

HOW TO RUN
----------
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/make_pilot_heatmaps.py
    # CPU (single small model, fine for inspection):
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/make_pilot_heatmaps.py --device cpu
    # Subset / different reference seed:
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/make_pilot_heatmaps.py --seeds 42-44 --reference-seed 43
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from visualization.attention_heatmaps import ADAPTERS
from compare_heatmaps import (
    align_heads_to_reference,
    plot_grid,
    _file_suffix,
    parse_seeds,
)

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
ARCH        = "vaswani_rope"

# label -> the prompt-name subfolder; hf is the gold head-final reorder of hi.
PILOT_PROMPTS = [
    {"label": "depth0_host_no_patch_1",
     "hi": "the boy likes a girl",  "hf": "the boy a girl likes"},
    {"label": "depth0_pathogen_no_patch_2",
     "hi": "a cat chases this dog", "hf": "a cat this dog chases"},
]


def last_checkpoint(seed: int) -> Path:
    seed_dir = RESULTS_DIR / ARCH / f"seed_{seed}"
    ckpts = [d for d in seed_dir.glob("checkpoint-*") if d.is_dir()]
    if not ckpts:
        sys.exit(f"no checkpoints under {seed_dir}")
    return max(ckpts, key=lambda d: int(d.name.split("-")[1]))


def word_id_map(hi: str) -> dict[str, str]:
    """0-based positional ID per HI word: 'the boy likes a girl' ->
    {'the':'0','boy':'1','likes':'2','a':'3','girl':'4'}.

    Keyed by surface word so the SAME word gets the SAME tag on both the HI and
    HF axes, regardless of where it lands after the head-final reorder.  (These
    pilot prompts have all-distinct words; a repeated word would share one ID.)
    """
    return {w: str(i) for i, w in enumerate(hi.split())}


def label_tokens(tokens: list[str], id_map: dict[str, str]) -> list[str]:
    """Tag each axis token with its HI ID ('boy' -> 'boy_1'); specials
    (<bos>/<eos>/<pad>/<sot>) and anything not in id_map stay bare."""
    return [f"{t}_{id_map[t]}" if t in id_map else t for t in tokens]


def render_prompt(label, hi, bundles, seeds, ref_idx, align) -> None:
    """One prompt's bundles (one per seed) -> per (family, layer) grids + CSV."""
    families = list(bundles[0].groups.keys())
    n_layers = bundles[0].groups[families[0]].shape[0]
    n_seeds  = len(seeds)
    id_map   = word_id_map(hi)

    # Fixed gold hf guarantees identical shapes; verify and fail loudly if not.
    for family in families:
        ref_shape = bundles[0].groups[family].shape
        for seed, b in zip(seeds, bundles):
            if b.groups[family].shape != ref_shape:
                sys.exit(
                    f"shape mismatch [{label}] seed {seed} family '{family}': "
                    f"{tuple(b.groups[family].shape)} != {tuple(ref_shape)}"
                )

    out_root = FIGURES_DIR / ARCH / "heatmaps" / label
    out_root.mkdir(parents=True, exist_ok=True)
    nonref = [si for si in range(n_seeds) if si != ref_idx]
    csv_rows: list[tuple] = []

    for family in families:
        per_seed = [b.groups[family] for b in bundles]      # list of [L, H, Tq, Tk]
        n_heads  = per_seed[0].shape[1]
        qtok     = label_tokens(bundles[0].query_tokens[family], id_map)
        ktok     = label_tokens(bundles[0].key_tokens[family], id_map)
        for layer in range(n_layers):
            vecs = [per_seed[s][layer].reshape(n_heads, -1) for s in range(n_seeds)]
            perms, matched, naive = align_heads_to_reference(vecs, ref_idx)

            oi = sum(sum(matched[s]) / n_heads for s in nonref) / max(len(nonref), 1)
            nv = sum(naive[s] for s in nonref) / max(len(nonref), 1)
            for s in range(n_seeds):
                csv_rows.append((family, layer, seeds[s], seeds[ref_idx],
                                 sum(matched[s]) / n_heads, naive[s]))

            save_path = out_root / f"layer_{layer}_{_file_suffix(family)}.png"
            title = (
                f"{ARCH} | {label} | {family} | layer {layer} | "
                f"rows=seeds cols=heads | order-inv cos={oi:.3f} (naive {nv:.3f})"
            )
            plot_grid(per_seed, seeds, layer, qtok, ktok, title, save_path,
                      perms, matched, ref_idx, align)
            print(f"  wrote {save_path.relative_to(EXP_DIR)}  "
                  f"(order-inv={oi:.3f}, naive={nv:.3f})", flush=True)

    csv_path = RESULTS_DIR / ARCH / f"{label}_heatmap_consistency.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "layer", "seed", "ref_seed",
                    "mean_matched_cosine", "mean_naive_cosine"])
        for family, layer, seed, ref_seed, mm, mn in csv_rows:
            w.writerow([family, layer, seed, ref_seed, f"{mm:.4f}", f"{mn:.4f}"])
    print(f"  metrics -> {csv_path.relative_to(EXP_DIR)} ({len(csv_rows)} rows)",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--align-heads", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reference-seed", type=int, default=None)
    p.add_argument("--device", default="cuda", help="cpu | cuda (default: cuda)")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    ref_idx = 0
    if args.reference_seed is not None:
        if args.reference_seed not in seeds:
            sys.exit(f"--reference-seed {args.reference_seed} not in --seeds {seeds}.")
        ref_idx = seeds.index(args.reference_seed)

    load, extract = ADAPTERS[ARCH]["load"], ADAPTERS[ARCH]["extract"]

    # Load each seed ONCE, extract both prompts, then free the model.
    per_label = {p["label"]: [] for p in PILOT_PROMPTS}
    for seed in seeds:
        ckpt = last_checkpoint(seed)
        model, tok = load(str(ckpt), args.device)
        for prompt in PILOT_PROMPTS:
            bundle = extract(model, tok, prompt["hi"], prompt["hf"], args.device, 0)
            per_label[prompt["label"]].append(bundle)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"[extract] seed {seed} ({ckpt.name})", flush=True)

    for prompt in PILOT_PROMPTS:
        print(f"\n=== {prompt['label']}  HI={prompt['hi']!r}  HF={prompt['hf']!r} ===",
              flush=True)
        render_prompt(prompt["label"], prompt["hi"], per_label[prompt["label"]],
                      seeds, ref_idx, args.align_heads)

    print("\nPilot heatmaps complete.", flush=True)


if __name__ == "__main__":
    main()
