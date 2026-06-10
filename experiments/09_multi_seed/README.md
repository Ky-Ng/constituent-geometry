# 09_multi_seed — Observations

## Goal
Check whether the attention heatmaps from experiments **06/07/08** (each a single
seed-0 model) are **representative across seeds**, or an artifact of one random
initialization. We re-run the same three trainings across **10 seeds (42–51)** on
the **same `random_frame` depth-3 dataset** and save all checkpoints locally, so a
follow-up step can compare heatmaps visually and quantitatively (cosine similarity)
across seeds.

> **Scope note.** This experiment covers **training only** — producing the 30
> checkpoints. The cross-seed heatmap comparison is **deferred** (see *Next step*).
> Grokking is also deferred: we keep **epochs = 30** (the 06/07/08 baseline) so the
> heatmaps stay directly comparable to the existing single-model figures. On
> `random_frame` depth-3 these models already reach ~1.0 test exact-match, so there
> is no generalization gap to "grok" here anyway — that question belongs on the
> `held_out_depth` datasets in a later experiment.

## Setup
- **Datasets/architectures:** reuses the existing scripts unchanged, one source of truth:
  | arch key | reused script | model |
  |---|---|---|
  | `vaswani` | `06_.../run.py` | Vaswani encoder-decoder, sinusoidal PE |
  | `vaswani_rope` | `07_.../run.py` | Vaswani encoder-decoder, RoPE on Q/K |
  | `gpt2_rope` | `08_.../run.py` | GPT2-style decoder-only, RoPE |
- **Dataset:** `kylelovesllms/hi_hf_frames_d3_random_100` (`random_frame`, depth 3).
- **Seeds:** 42–51 (10). **Epochs:** 30. All other knobs (tokenizer, max-length,
  optimizer, eval) stay at each script's own defaults.
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/seed_<S>/best/` — we do **not** push 30 models to the HF Hub.

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out to
the 06/07/08 scripts with a different `--seed` each; it never redefines model or
training logic.

## How to run

**Default — packed (3 jobs, one per architecture, all 10 seeds share one GPU):**
```bash
sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/09_multi_seed/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 30 tasks):**
```bash
sbatch --array=0-29 slurm/run_gpu.sbatch experiments/09_multi_seed/run.py
# task -> (arch, seed):  arch = ARCHS[task // 10], seed = 42 + task % 10
```

**Local smoke test (2 seeds, 1 epoch, no wandb / no push):**
```bash
uv run python experiments/09_multi_seed/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/09_multi_seed_<arch>_seed_<S>.log`; wandb runs are
named `09_multi_seed_<arch>_seed_<S>` (disable with `--no-wandb`).

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer's wandb integration — **no edits to 06/07/08**):
- `WANDB_RUN_GROUP = 09_multi_seed_<arch>` — the 10 seeds of each architecture
  share a **group**, so the wandb UI collapses them into one panel with a **mean
  curve + min/max band across seeds**. That band *is* the cross-seed-consistency
  signal for the metric curves (e.g. eval `exact_match`).
- `WANDB_TAGS = 09_multi_seed,<arch>,seed_<S>` — filter all 30 runs at once with
  the `09_multi_seed` tag.

In the wandb UI (project `constituent-geometry`):
1. Filter runs by tag `09_multi_seed` to isolate this experiment's 30 runs.
2. The three groups (`09_multi_seed_vaswani`, `_vaswani_rope`, `_gpt2_rope`) each
   show a mean ± band; multi-select them to **overlay the three architectures** on
   one chart, or expand a group to see the individual per-seed lines.
3. Toggle **"Group by"** off to see all 30 runs as separate lines if you prefer.

## Throughput — why packed is the default
These models are tiny (~1.8M params, batch 64, seq ≤64): weights ≈7 MB, optimizer
state + grads ≈22 MB, activations a few hundred MB, plus a ~0.5 GB CUDA context →
**well under 1 GB** of a 48 GB A6000. A single run barely loads the GPU.

- **Memory:** batch 64 is nowhere near saturating GPU memory; ~10 concurrent runs ≈
  5–10 GB, comfortably within 48 GB.
- **Time-sharing:** each subprocess has its own CUDA context. Without MPS the driver
  time-slices kernels across contexts, but any single run is latency/CPU-bound (lots
  of Python + data loading between tiny kernels), so the GPU is idle most of the time
  for one run — interleaving ~10 fills those gaps (~4–7× throughput). With NVIDIA MPS
  the contexts overlap on SM partitions for even better utilization.

| | Packed (default, 3 jobs) | Array (`--array=0-29`, 1 GPU/run) |
|---|---|---|
| GPUs used | 3 | up to 30 |
| Per-GPU use | ~5–10 GB / 48 GB | ~1 GB / 48 GB (very low) |
| Queueing | needs 3 free GPUs | needs up to 30 free, else queues in waves |
| Wall-clock | a few minutes | ~one run if 30 GPUs free |

Packed uses 3 GPUs instead of 30 for jobs that each use <1 GB — polite to the
cluster and finishes in minutes. Use array when you'd rather trade GPUs for clean
per-task logs and the lowest wall-clock on an idle cluster.

## Results

_No results yet._ After the sweep, expect 30 checkpoints at
`results/<arch>/seed_<S>/best/` (3 archs × 10 seeds).

## Cross-seed heatmap consistency (`compare_heatmaps.py`)
For one architecture and one **fixed prompt**, render — per attention family, per
layer — an **S × H grid** (rows = seeds, columns = heads) so you can eyeball whether
the seeds learned the same attention pattern. Reuses
`src/visualization/attention_heatmaps.py` (`ADAPTERS`) for load + extraction; no
model logic duplicated.

- A **fixed gold `(hi, hf)`** pair is teacher-forced for every seed (default: a row
  of the dataset's `test` split), so all grids share identical token axes and tensor
  shapes `[L, H, Tq, Tk]` and are directly comparable.
- Heatmaps render **square** via `ax.set_box_aspect(1)` (holds even for
  cross-attention where `Tq ≠ Tk`).
- **Encoder-decoder** models (`vaswani`, `vaswani_rope`) produce all three families
  (`encoder_self`, `decoder_self`, `cross`); `gpt2_rope` produces one (`self`).

By default the columns are **reference-aligned** (see *How head matching works*),
so each seed's heads are reordered to best-match the first seed; pass
`--no-align-heads` for the raw head order. The script also writes a
**permutation-invariant cosine** to `results/<arch>/heatmap_consistency.csv`.

**Outputs:**
- Figures: `figures/<arch>/heatmaps/[<prompt-name>/]layer_<l>_<suffix>.png`, where
  `<suffix>` is `attention` for the single-family `gpt2_rope`, else the family name.
  So `vaswani` (L=4, H=4, S=10) yields 4 layers × 3 families = **12 images**, each a
  10 × 4 grid; `gpt2_rope` yields **4 images** named `layer_<l>_attention.png`.
- Numbers: `results/<arch>/heatmap_consistency.csv` — one row per (family, layer,
  seed) with `mean_matched_cosine` (order-invariant) vs `mean_naive_cosine`
  (same-index). The gap is the head-permutation effect.

```bash
# default gold pair = test-split row 0; columns reference-aligned
uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani
uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani_rope --split test --index 3
uv run python experiments/09_multi_seed/compare_heatmaps.py --arch vaswani --no-align-heads   # raw order
# explicit prompt instead of a dataset row:
uv run python experiments/09_multi_seed/compare_heatmaps.py --arch gpt2_rope \
    --hi "the dancer thinks that a cat knows Betty" --hf "<gold hf string>"
```

### Depth-stratified sweep over a fixed prompt set (two-stage, CPU)
`compare_heatmaps.py` above handles **one** prompt and loads every model itself. To
study how consistency varies with embedding **depth**, we fix **3 prompts at each of
depths 0–3** (12 total; see `prompts.py`) and run them across all seeds. Because the
expensive part — load a model + forward — is per `(arch, seed)` and embarrassingly
parallel, but a single grid needs *all* seeds together, the work is **split in two**:

1. **`extract_attention.py` (Stage 1, the parallel part).** One CPU job per
   `(arch, seed)` loads that model **once**, forwards all 12 prompts, and caches each
   `AttentionBundle` to `results/<arch>/seed_<S>/attn/depth<D>-ex<N>.pt`. 3 archs ×
   10 seeds = **30 independent jobs** (`SLURM_ARRAY_TASK_ID → arch = ARCHS[task // 10]`,
   `seed = seeds[task % 10]`).
2. **`gather_heatmaps.py` (Stage 2, cheap).** Reads the cached bundles (no models),
   renders the reference-aligned **S × H** grids, and writes one consistency CSV per
   arch. Single CPU job over all archs × prompts.

The matching math (`align_heads_to_reference`) and the renderer (`plot_grid`) are
**imported from `compare_heatmaps.py`** — one source of truth; Stage 2 just feeds it
cached tensors instead of freshly-extracted ones. The cache is also the substrate for
the planned **similarity-metric** iteration (another cheap pass over the `.pt` files,
no re-inference).

```bash
# Stage 1 — 30-task CPU array (the per-seed parallelism). No GPU needed.
sbatch --array=0-29 slurm/run_cpu.sbatch experiments/09_multi_seed/extract_attention.py

# Stage 2 — once Stage 1's caches exist (single CPU job, or just run locally).
sbatch slurm/run_cpu.sbatch experiments/09_multi_seed/gather_heatmaps.py
# uv run python experiments/09_multi_seed/gather_heatmaps.py            # local equivalent
```

**Outputs:** `figures/<arch>/heatmaps/depth<D>-ex<N>/layer_<L>_<type>.png` (columns
cosine-reordered, rows = the 10 seeds) and `results/<arch>/heatmap_consistency.csv`
with `depth, ex, family, layer, seed, ref_seed, mean_matched_cosine,
mean_naive_cosine`.

### How head matching works
Attention heads within a layer have **no canonical order**: permuting the heads
(and the matching slices of the projection weights) gives a *functionally identical*
model. So comparing "seed 42 head 0" to "seed 43 head 0" compares arbitrary labels —
two seeds can be equivalent yet a column won't line up.

For each (family, layer) we instead **match** every seed's heads to the reference
seed (the first, seed 42) by the one-to-one pairing that **maximizes total cosine
similarity** (the assignment problem). With only `H=4` heads we solve it *exactly* by
trying all `4! = 24` pairings — no `scipy` needed. The matched pairing is used two
ways: (1) reorder that seed's **columns** so column *j* shows the head matching
reference head *j*; (2) the **average matched cosine** is the order-invariant
consistency score we report.

**Worked example** (3-head toy, one family/layer; each head's attention map flattened
to a vector). Reference = seed 42, compared to seed 43:

- seed 42 heads: `h0=[1,0,0]`, `h1=[0,1,0]`, `h2=[0,0,1]`
- seed 43 heads: `h0=[0,0.9,0.1]`, `h1=[0.1,0,1]`, `h2=[1,0.1,0]`

Cosine matrix (rows = seed-42 heads, cols = seed-43 heads):

```
            43-h0   43-h1   43-h2
42-h0       0.00    0.10    0.99
42-h1       0.99    0.00    0.10
42-h2       0.11    0.99    0.00
```

- **Naive same-index** = diagonal mean = (0.00+0.00+0.00)/3 = **0.00** → looks like the seeds disagree completely.
- **Order-invariant** = best of the 3! = 6 pairings → `42-h0→43-h2`, `42-h1→43-h0`,
  `42-h2→43-h1`, i.e. `perm = [2,0,1]`, each ≈ 0.99 → mean **≈ 0.99**. The seeds
  learned the *same* heads, just relabeled.

So the grid reorders seed 43's columns to `[h2, h0, h1]` and reports 0.99, not 0.00.
The matched-vs-naive gap in the CSV is exactly this head-permutation effect.

### Is matching done only on the final layer?
**No — it is recomputed independently for every (family, layer).** A head in
encoder layer 0 is a different module from a head in encoder layer 1 (and from any
decoder-self / cross head), so each (family, layer) gets its **own** `H×H` matching
and its **own** permutation. For `vaswani` that's `3 families × 4 layers = 12`
separate matchings; each of the 12 images is aligned on its own. A seed's head
ordering in `encoder_self` layer 0 can therefore differ from its ordering in
`cross` layer 2 — which is correct, since those heads are unrelated.

> **Interpretation caveat:** matching is computed on a **single prompt**, so a head's
> identity is inferred from one input. Aggregating each head's map over several
> prompts before matching would give a more robust identity (a natural extension).

## Notes
-
