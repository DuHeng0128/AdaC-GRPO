# Training

Use Linux, Python 3.10/3.11 and CUDA. The main configuration uses
Qwen2.5-VL-7B-Instruct on four 80 GB A800 GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
python -m pip install 'flash-attn>=2.7,<3' --no-build-isolation
python scripts/validate_data.py
```

Install the PyTorch build appropriate for your CUDA environment. The custom
trainer uses the TRL 0.16 / DeepSpeed 0.15 API family specified in `setup.py`.
Dataset tools and prediction scoring do not require GPU dependencies.

## SFT warm-up

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR="$PWD/outputs/sft" \
bash scripts/run_sft_video.sh
```

The launcher uses the 300 demonstrations in `data/sudoku/sft.json`, ZeRO-2,
learning rate 1e-6, and three epochs (`SFT_EPOCHS` can override this default).

## Adaptive curriculum GRPO

```bash
MODEL_PATH="$PWD/outputs/sft" \
OUTPUT_DIR="$PWD/outputs/adac-grpo" \
bash scripts/run_grpo_video.sh
```

| Setting | Default |
| --- | --- |
| Precision / parallelism | bfloat16 / ZeRO-3 |
| Learning rate / scheduler | 1e-6 / cosine |
| Weight decay / gradient norm | 0.01 / 5.0 |
| Per-device batch / accumulation | 1 / 1 |
| Rollouts / KL beta | 8 / 0.04 |
| Prompt / completion tokens | 16,384 / 768 |
| Image pixel bounds | 3,136–401,408 |
| Epochs per selected subset | 3 |
| ZPD / selection | [0.3, 0.8] / argmax |
| Online evaluation | 1/4 of each active training tier; no extra cap |
| Repeated-subset stopping | 3 |

GRPO settings follow [paper Appendix B](https://arxiv.org/html/2610.13184v1#A2).
Each curriculum call trains the selected subset and starts its own cosine
schedule. A tier remains eligible until its online accuracy exceeds 0.8;
evaluation never uses the held-out test split. Checkpoints are saved under
`model_A`, `model_B`, and `model_C` in the output directory.

Both launchers accept `NPROC_PER_NODE`, `MASTER_PORT`, and additional Trainer
arguments. Use `--use_curriculum false` for standard GRPO, or
`--curriculum_selection_strategy median` / `random` for selection ablations.
The paper's selection and threshold ablations use one epoch; set both
`--epochs_per_subset 1 --num_train_epochs 1` for those runs.

## Evaluation

Generate one response per test image using the prompts in paper Appendix D,
then score saved responses with [`evaluate_predictions.py`](../scripts/evaluate_predictions.py).
The [dataset guide](dataset.md) describes the prediction format.
