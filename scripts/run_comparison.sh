#!/usr/bin/env bash
# Runs three curriculum-learning experiments back-to-back, one per selection
# strategy (argmax / median / random).  Each writes its own log and model
# output so results can be compared afterwards.
#
# Usage:
#   bash run_comparison.sh                  # run all three
#   bash run_comparison.sh argmax median    # run a subset

set -euo pipefail

cd "$(dirname "$0")/../r1-v"

# ── Shared env ────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export DEBUG_MODE="true"
export WANDB_DISABLED=true

# ── Shared training args ──────────────────────────────────────────────────────
MODEL_PATH="./src/r1-v/log/Qwen2.5-VL-7B-cot-sft-ALL"
DATASET="./sudoku/puzzles.json"
BASE_OUTPUT_DIR="/path/to/output/Selection"
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

GPUS="0,1,2,3"
NPROC=4
MASTER_PORT=12365

# Strategies to run (override via CLI args, e.g. bash run_comparison.sh argmax)
STRATEGIES=("${@:-argmax median random}")
# Re-split if passed as a single string (no args case)
if [ "${#@}" -eq 0 ]; then
    STRATEGIES=(median random)
fi

# ── Per-strategy loop ─────────────────────────────────────────────────────────
for STRATEGY in "${STRATEGIES[@]}"; do
    # Capitalise first letter for directory / run-name aesthetics
    STRATEGY_CAP="$(tr '[:lower:]' '[:upper:]' <<< "${STRATEGY:0:1}")${STRATEGY:1}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${STRATEGY_CAP}"
    LOG_FILE="${LOG_DIR}/${STRATEGY}.log"
    RUN_NAME="Video-R1-C-${STRATEGY_CAP}"

    export LOG_PATH="${OUTPUT_DIR}/debug_log.txt"
    mkdir -p "${OUTPUT_DIR}"

    echo "================================================================"
    echo "  Starting strategy: ${STRATEGY}  →  ${OUTPUT_DIR}"
    echo "  Log: ${LOG_FILE}"
    echo "  $(date)"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
      --nproc_per_node="${NPROC}" \
      --nnodes="1" \
      --node_rank="0" \
      --master_addr="127.0.0.1" \
      --master_port="${MASTER_PORT}" \
      src/open_r1/grpo.py \
      --output_dir "${OUTPUT_DIR}" \
      --model_name_or_path "${MODEL_PATH}" \
      --dataset_name "${DATASET}" \
      --torch_dtype "bfloat16" \
      --deepspeed local_scripts/zero3.json \
      --max_prompt_length 16384 \
      --max_completion_length 768 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-6 \
      --lr_scheduler_type "cosine" \
      --weight_decay 0.01 \
      --bf16 \
      --logging_steps 1 \
      --gradient_checkpointing true \
      --temporal false \
      --len_control true \
      --attn_implementation flash_attention_2 \
      --max_pixels 401408 \
      --num_train_epochs 1 \
      --epochs_per_subset 1 \
      --eval_fixed_size 64 \
      --run_name "${RUN_NAME}" \
      --save_steps 10000 \
      --beta 0.04 \
      --max_grad_norm 5 \
      --save_only_model true \
      --num_generations 8 \
      --use_curriculum true \
      --per_device_eval_batch_size 32 \
      --accuracy_threshold_for_switch 0.15 \
      --accuracy_threshold_upper 0.8 \
      --curriculum_selection_strategy "${STRATEGY}" \
      --save_total_limit 1 \
      2>&1 | tee -a "${LOG_FILE}"

    echo "  Finished ${STRATEGY} at $(date)"
done

echo "================================================================"
echo "  All experiments complete."
echo "  Outputs : ${BASE_OUTPUT_DIR}/{Argmax,Median,Random}/"
echo "  Logs    : ${LOG_DIR}/"
echo "================================================================"
