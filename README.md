# AdaC-GRPO: Adaptive Curriculum GRPO for Visual Reasoning

This repository contains the code and dataset for training vision-language models with Adaptive Curriculum GRPO (AdaC-GRPO) on visual Sudoku puzzles.

```
AdaC-GRPO/
├── src/open_r1/          Core training code
│   ├── grpo.py           Main GRPO training script with AdaC-GRPO support
│   ├── sft_video.py      SFT warm-up script
│   ├── evaluate.py       Evaluation utilities
│   ├── generate.py       Generation utilities
│   └── trainer/          Custom GRPO trainers
│       ├── grpo_trainer.py
│       └── vllm_grpo_trainer_modified.py
├── configs/              DeepSpeed / accelerate configs
│   ├── zero2.yaml
│   ├── zero3.yaml
│   ├── ddp.yaml
│   └── deepspeed/        DeepSpeed JSON configs
├── scripts/              Launch scripts
├── data/                 Sudoku dataset
│   ├── sudoku
│   ├── sudoku_test
├── setup.py
└── setup.cfg
```

---

## Method

AdaC-GRPO extends GRPO with an adaptive curriculum that dynamically selects which difficulty subset to train on at each step, based on the model's current accuracy on each subset.

**Difficulty levels:** 12 subsets (A1–A4, B1–B4, C1–C4), ordered from easiest to hardest.

**Selection strategy:** At each curriculum step, evaluate a sample of each undone subset and select the one with the highest accuracy above a threshold (`accuracy_threshold_for_switch`). If no subset qualifies, continue training the current subset.

**Key hyperparameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `use_curriculum` | `false` | Enable AdaC-GRPO curriculum |
| `curriculum_selection_strategy` | `argmax` | Subset selection: `argmax` or `random` |
| `accuracy_threshold_for_switch` | `0.1` | Min accuracy to switch to a new subset |
| `accuracy_threshold_upper` | `0.8` | Max accuracy — skip subsets above this |
| `epochs_per_subset` | `1` | Training epochs per selected subset |
| `eval_fixed_size` | `64` | Number of samples for subset evaluation |
| `temporal` | `false` | Use temporal GRPO (video-specific) |
| `len_control` | `true` | Enable length-control reward |
| `beta` | `0.04` | KL penalty coefficient |
| `num_generations` | `8` | Rollouts per prompt |

---

## Dataset

The dataset consists of visual Sudoku puzzles at three grid sizes (4×4, 6×6, 9×9), rendered as PNG images.

### `data/sudoku/` — Training data

| File | Size | Description |
|------|------|-------------|
| `grpo_A.json` | 1,024 | Difficulty group A (easiest) |
| `grpo_B.json` | 1,024 | Difficulty group B (medium) |
| `grpo_C.json` | 1,024 | Difficulty group C (hardest) |
| `sft.json` | 300 | SFT warm-up data with chain-of-thought |
| `puzzles.json` | 3,072 | Full puzzle metadata (image path → grid) |
| `4x4/`, `6x6/`, `9x9/` | 6,129 images | Puzzle images by grid size |

Each GRPO entry contains:
```json
{
  "problem_id": 1,
  "problem": "Please read this 6x6 puzzle ...",
  "data_type": "image",
  "problem_type": "sudoku",
  "solution": "<answer>[[...]]</answer>",
  "path": "6x6/puzzle/puzzle_006_00001.png"
}
```

### `data/sudoku_base/` — Full puzzle base

18,000 puzzle images across all grid sizes, used for generating additional training data.

---

## Installation

```bash
pip install -e .
pip install torch transformers trl deepspeed flash-attn --no-build-isolation
pip install nltk rouge-score qwen-vl-utils
```

---

## Training

### Step 1: SFT warm-up

Fine-tune the base model on chain-of-thought demonstrations before GRPO:

```bash
bash scripts/run_sft_video.sh
```

Key arguments:
```bash
python src/open_r1/sft_video.py \
    --model_name_or_path <model_path>/Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name data/sudoku/sft.json \
    --output_dir ./output/sft \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --deepspeed configs/deepspeed/zero2.json \
    --bf16 --gradient_checkpointing true \
    --attn_implementation flash_attention_2
```

### Step 2: AdaC-GRPO training

```bash
bash scripts/run_grpo_video.sh
```

Key arguments:
```bash
python src/open_r1/grpo.py \
    --model_name_or_path ./output/sft \
    --dataset_name data/sudoku/puzzles.json \
    --output_dir ./output/adac_grpo \
    --use_curriculum true \
    --curriculum_selection_strategy argmax \
    --accuracy_threshold_for_switch 0.15 \
    --accuracy_threshold_upper 0.8 \
    --epochs_per_subset 1 \
    --eval_fixed_size 64 \
    --num_generations 8 \
    --beta 0.04 \
    --learning_rate 1e-6 \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --deepspeed configs/deepspeed/zero3.json \
    --bf16 --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408
```

### Baseline (standard GRPO, no curriculum)

```bash
python src/open_r1/grpo.py \
    --use_curriculum false \
    --dataset_name data/sudoku/grpo.json \
    ... (same other args)
```

---

## Evaluation

```bash
python src/open_r1/evaluate.py \
    --model_name_or_path ./output/adac_grpo \
    --dataset_name data/sudoku/grpo.json
```

---

## Hardware Requirements

| Config | GPUs | Parallelism |
|--------|------|-------------|
| Qwen2.5-VL-7B | 4 × A100/A800 (80GB) | DeepSpeed ZeRO-3 |
| Qwen2.5-VL-7B (single GPU debug) | 1 × A100 | DeepSpeed ZeRO-3 |

---

## Dependencies

```
torch >= 2.5
transformers >= 4.50
trl >= 0.12
deepspeed >= 0.15
flash-attn >= 2.7
qwen-vl-utils
nltk
rouge-score
```
