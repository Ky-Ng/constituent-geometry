# Experiments

Each numbered folder is a self-contained experiment with its own `run.py`, results, and logs.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 00 | `00_example` | Template | End-to-end sanity check of the project setup |
| 01 | `01_build_simple_hi_hf_ds` | Done | Enumerate the depth-3 HI/HF toy-CFG dataset |
| 02 | `02_train_vaswani_simple_hi_hf_ds_depth_3` | Done (caveats) | Train on a random split; ~0.003 loss, but i.i.d. eval can't show generalization |
| 03 | `03_holdout_depth_2_vaswani_hi_hf` | Done | Depth-holdout (train {0,1}, test depth 2), shrunk model (1 head / 2 layers) — underfit |
| 04 | `04_holdout_depth_2_vaswani_hi_hf_heads_4_layers_4` | Done | Capacity control for 03 (4 heads / 4 layers, same small width). Solves depth {0,1} (exact-match 1.0) but fails to generalize to depth 2 (exact-match 0.0, token-acc 0.47) — learned a non-recursive solution |
| 05 | `05_build_dataset_with_frames` | Ready | Frame-based HI/HF dataset builder for the new sub-categorized grammar (~3.3B sentences at depth 3 was infeasible; this enumerates only 90 frames and samples K lexical fillings each). Supports `random_frame` and `held_out_depth` structural splits over frames, not rows. |
| 06 | `06_vaswani_original_hi_hf_frames_heads_4_layers_4_random` | Ready | First training run on the frame-based dataset (experiment 05, `random_frame` split). Original Vaswani encoder-decoder at paper-ish width (`d_model=128, d_ff=512`), shape `n_heads=4, n_layers=4`. Loads a Hub dataset id (default `kylelovesllms/hi_hf_frames_d3_random_100`). |
| 07 | `07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4` | Ready | One-knob ablation of 06: same data / width / shape / optimizer, but the fixed sinusoidal positional table is replaced by RoPE applied to Q/K inside self-attention (cross-attention un-rotated). Same three runs as 06 (`random_depth_3`, `heldoutdepth_3`, `heldoutdepth_4`). |

## Convention
- Create new numbered folders (`01_xxx/`, `02_xxx/`, ...) for new experiments — don't edit old ones.
- Each folder contains: `run.py`, `results/`, `logs/`, `figures/`, and `README.md` (observations).
- This README should only contain brief descriptions of each experiment. Detailed setup, results, and observations belong in each experiment's own `README.md`.
- The `artifacts/` folder should contain large blob files (e.g. `.safetensors` and `.pt` files which should be uploaded to HuggingFace instead of git)
- During a new experiment, you should `cp` the `00_example` folder and rewrite over it