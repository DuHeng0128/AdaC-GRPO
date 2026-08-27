cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_sft.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "/path/to/output/Qwen3-VL-8B-cot-sft-ALL" \
    --model_name_or_path "<model_path>/Qwen/Qwen3-VL-8B-Instruct" \
    --dataset_name "sudoku/sft.json" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name Qwen3-VL-8B-Video-cot-sft \
    --save_steps 400 \
    --max_grad_norm 5 \
    --save_only_model true \