#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/sft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="${MASTER_PORT:-29500}" --module open_r1.sft_video \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$REPO_ROOT/data/sudoku/sft.json" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$REPO_ROOT/configs/deepspeed/zero2.json" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 --num_train_epochs "${SFT_EPOCHS:-3}" \
  --torch_dtype bfloat16 --bf16 --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --logging_steps 1 --save_steps 400 --save_total_limit 1 \
  --max_grad_norm 5 --save_only_model true --report_to none --seed 42 "$@"
