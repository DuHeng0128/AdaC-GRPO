#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the SFT warm-start checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/adac-grpo}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="${MASTER_PORT:-29501}" --module open_r1.grpo \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$REPO_ROOT/data/sudoku/puzzles.json" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$REPO_ROOT/configs/deepspeed/zero3.json" \
  --torch_dtype bfloat16 --bf16 \
  --gradient_checkpointing true --attn_implementation flash_attention_2 \
  --use_curriculum true --curriculum_selection_strategy argmax \
  --accuracy_threshold_for_switch 0.3 --accuracy_threshold_upper 0.8 \
  --eval_sample_ratio 0.25 --eval_fixed_size 0 \
  --epochs_per_subset 3 --num_train_epochs 3 --early_stop_same_repeat 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 1e-6 --lr_scheduler_type cosine --weight_decay 0.01 \
  --max_grad_norm 5 --beta 0.04 --num_generations 8 \
  --max_prompt_length 16384 --max_completion_length 768 \
  --max_pixels 401408 --min_pixels 3136 \
  --reward_funcs accuracy format --temporal false --len_control true \
  --logging_steps 1 --save_steps 10000 --save_total_limit 1 \
  --save_only_model true --report_to none --seed 42 "$@"
