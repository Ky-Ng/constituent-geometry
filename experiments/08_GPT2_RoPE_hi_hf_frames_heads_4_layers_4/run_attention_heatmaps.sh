#!/usr/bin/env bash
# Run attention-heatmap visualization for all three model families across
# three example prompts at each of depths 3, 2, 1.
# Output: experiments/<exp>/figures/depth<D>-ex<N>/
set -euo pipefail
cd "$(dirname "$0")/.."

# Depth-indexed prompt arrays (3 examples each).
declare -a D3=(
  "the boy likes that Iskarous assumes that a friend knows that this student chases Betty"
  "this engineer knows that James likes that a sister hates that a student loves Iskarous"
  "a teacher thinks that Mary assumes that the dog thinks that this father pursues Hamilton"
)
declare -a D2=(
  "this dog knows that Jia claims that Iskarous dances"
  "the artist claims that Jia believes that Jia applauds"
  "the boy hates that Shri claims that Mary sings"
)
declare -a D1=(
  "the dancer thinks that a cat knows Betty"
  "a cat assumes that this teacher pursues James"
  "a musician thinks that this artist believes Jia"
)

# model_type|model_repo|experiment_figures_dir
declare -a MODELS=(
  "vaswani|kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3|experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/figures"
  "vaswani_rope|kylelovesllms/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3|experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/figures"
  "gpt2_rope|kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3|experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/figures"
)

run_one() {
  local mtype="$1" model="$2" outroot="$3" depth="$4" exnum="$5" prompt="$6"
  local outdir="${outroot}/depth${depth}-ex${exnum}"
  echo "=== [${mtype}] depth${depth}-ex${exnum} -> ${outdir}"
  uv run python -m visualization.attention_heatmaps \
    --model-type "$mtype" \
    --model "$model" \
    --hi "$prompt" \
    --output-dir "$outdir"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r mtype model outroot <<< "$entry"
  echo "##### MODEL: $model ($mtype) #####"
  for i in 0 1 2; do run_one "$mtype" "$model" "$outroot" 3 $((i+1)) "${D3[$i]}"; done
  for i in 0 1 2; do run_one "$mtype" "$model" "$outroot" 2 $((i+1)) "${D2[$i]}"; done
  for i in 0 1 2; do run_one "$mtype" "$model" "$outroot" 1 $((i+1)) "${D1[$i]}"; done
done

echo "ALL DONE"
