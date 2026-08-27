cd src/r1-v

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log.txt"
export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="12365" \
  src/open_r1/grpo.py \
  --output_dir "/path/to/output/Selection/Argmax" \
  --model_name_or_path "/path/to/log/Qwen2.5-VL-7B-cot-sft-ALL" \
  --dataset_name "./sudoku/puzzles.json" \
  --torch_dtype "bfloat16" \
  --deepspeed local_scripts/zero3.json \
  --max_prompt_length 16384 \
  --max_completion_length 768 \
  --per_device_train_batch_size 1 \
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
  --run_name Video-R1-C \
  --save_steps 10000 \
  --beta 0.04 \
  --max_grad_norm 5 \
  --save_only_model true \
  --num_generations 8 \
  --use_curriculum false \
  --per_device_eval_batch_size 32 \
  --accuracy_threshold_for_switch 0.15 \
  --accuracy_threshold_upper 0.8 \
  --curriculum_selection_strategy "argmax" \
  --save_total_limit 1 \
#  --start_subset B1 \
#  2>&1 | tee -a /path/to/train.log
