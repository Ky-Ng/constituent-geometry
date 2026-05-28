#!/bin/bash
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# CPU-only template. Use for dataset building, analysis, plotting — anything
# that doesn't touch torch / CUDA. For GPU jobs use run_gpu.sbatch.
#
# Override resources on the command line as needed, e.g.:
#   sbatch --cpus-per-task=8 --mem=32G --time=08:00:00 \
#     slurm/run_cpu.sbatch experiments/05_build_dataset_with_frames/run.py \
#     --max-depth 3 --k 10000 --split-policy random_frame \
#     --out data/hi_hf_frames_d3_random
#
# Default usage:
#   sbatch slurm/run_cpu.sbatch experiments/05_build_dataset_with_frames/run.py \
#     --max-depth 2 --k 100 --split-policy random_frame \
#     --out data/hi_hf_frames_d2_random

mkdir -p slurm_logs

module purge
module load gcc/13.3.0

cd $SLURM_SUBMIT_DIR

uv run python "$@"
