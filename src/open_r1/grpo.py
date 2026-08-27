# train_curriculum_puzzle.py
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import gc
import os
import json
import re
import random
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import ast
import time

from datasets import Dataset as HFDataset, DatasetDict, load_dataset
import deepspeed
import torch.distributed as dist
from transformers import Qwen2VLForConditionalGeneration
import torch
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

difficulty_mapping = {
    'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3,
    'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7,
    'C1': 8, 'C2': 9, 'C3': 10, 'C4': 11
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )

    # Curriculum related
    use_curriculum: bool = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning (split dataset by difficulty and train subsets)."},
    )
    eval_sample_ratio: float = field(
        default=1,
        metadata={"help": "Fraction of each subset to sample for evaluation (default 1/16)."},
    )
    early_stop_same_repeat: int = field(
        default=3,
        metadata={"help": "Stop early if same subset trained this many times in a row."},
    )
    accuracy_threshold_for_switch: float = field(
        default=0.1,
        metadata={"help": "Only switch to an undone subset if its eval accuracy > this threshold."},
    )
    accuracy_threshold_for_double: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, and the next subset's eval accuracy is below this value "
                "(checked after accuracy_threshold_for_switch), the training epochs "
                "for that subset are doubled (epochs_per_subset * 2)."
            )
        },
    )
    eval_fixed_size: int = field(
        default=16,
        metadata={"help": "Maximum number of samples per subset used during curriculum evaluation."},
    )
    epochs_per_subset: int = field(
        default=1,
        metadata={"help": "Number of training epochs to run on each curriculum subset."},
    )
    start_subset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Label of the subset to start curriculum training from (e.g. 'B3'). "
                "All subsets with a lower difficulty index are marked as done and skipped. "
                "If not set, training starts from the first available subset (A1)."
            )
        },
    )
    accuracy_threshold_upper: float = field(
        default=0.8,
        metadata={"help": "Upper accuracy bound for candidate subsets during curriculum selection."},
    )
    curriculum_selection_strategy: str = field(
        default="argmax",
        metadata={
            "help": (
                "Strategy for picking the next subset from candidates within "
                "[accuracy_threshold_for_switch, accuracy_threshold_upper]. "
                "Choices: 'argmax' (highest accuracy), 'median' (median accuracy), "
                "'random' (uniform random)."
            )
        },
    )


def accuracy_reward(completions, solution, **kwargs):
    """
    Existing accuracy_reward extended with 'puzzle' question type handling:
    - if problem_type == 'puzzle' then compare grid outputs and return fraction of correct cells (0..1)
    Otherwise keep original behaviors.
    """
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        pattern2 = r'<answer>\s*(.*)'
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            # fallback
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    def sudoku_accuracy_reward(reference, hypothesis_matrix):
        if hypothesis_matrix is None:
            return 0.0
        
        def clean_sudoku_string(s):
            s = re.sub(r'[^0-9,\[\] ]', '', s)
            s = s.replace(" ", "") 
            s = s.replace(",,", ",")
            return s.strip()
        
        try:
            ref_clean = clean_sudoku_string(reference)
            hyp_clean = clean_sudoku_string(hypothesis_matrix)
            
            if not ref_clean.startswith("[[") or not ref_clean.endswith("]]"):
                return 0.0
            if not hyp_clean.startswith("[[") or not hyp_clean.endswith("]]"):
                return 0.0
                
            reference_list = ast.literal_eval(ref_clean)
            hypothesis_list = ast.literal_eval(hyp_clean)
        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Parsing error: {e}\nreference: {reference}\nhypothesis: {hypothesis_matrix}")
            return 0.0

        N = len(reference_list)
        if len(hypothesis_list) != N:
            return 0.0

        for i in range(N):
            if len(hypothesis_list[i]) != N:
                return 0.0
            for j in range(N):
                if reference_list[i][j] != hypothesis_list[i][j]:
                    return 0.0          
        return 1.0

    # problem_type may be passed as a list in kwargs (trainer original code used kwargs['problem_type'][0])
    question_type = None
    if 'problem_type' in kwargs:
        pt = kwargs['problem_type']
        if isinstance(pt, (list, tuple)) and len(pt) > 0:
            question_type = pt[0]
        elif isinstance(pt, str):
            question_type = pt

    if isinstance(completions[0], str):
        contents = completions
    else:
        contents = [completion[0]["content"] for completion in completions]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol) if isinstance(sol, str) else (sol if isinstance(sol, str) else "")
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
            elif question_type == "puzzle":
                # parse both grids and compute fraction of correct cells.
                reward = sudoku_accuracy_reward(gt_ans, output_ans)
                reward = max(0.0, min(1.0, reward))
                #print(f"Puzzle reward computed: {reward}")
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./debug_reward.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if isinstance(completions[0], str):
        contents = completions
    else:
        contents = [completion[0]["content"] for completion in completions]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []

    for content in contents:
        # Strip leading/trailing whitespace before matching.
        # Models commonly prepend a '\n' or append trailing text after </answer>;
        # re.fullmatch would fail on either, causing permanent 0 reward.
        stripped = content.strip()
        total_reward = 0.0
        format_match = re.fullmatch(pattern, stripped, re.DOTALL)
        format_reward = 1.0 if format_match else 0.0

        content_reward = 0.0
        if format_match:
            answer_match = re.search(r"<answer>(.*?)</answer>", stripped, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                if is_valid_2d_array(answer_content):
                    content_reward = 1.0

        total_reward = (format_reward * 0.5 + content_reward * 0.5)
        rewards.append(total_reward)

    return rewards

def is_valid_2d_array(content):
    try:
        parsed = ast.literal_eval(content)
        if not isinstance(parsed, list):
            return False
        for sublist in parsed:
            if not isinstance(sublist, list):
                return False
            for item in sublist:
                if not isinstance(item, (int, float)):
                    return False
        return True
    except (SyntaxError, ValueError):
        return False



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# --------------------------- Dataset helpers (解析你的 puzzle.json) ---------------------------

DEFAULT_LABEL_ORDER = [
    "A1","A2","A3","A4",
    "B1","B2","B3","B4",
    "C1","C2","C3","C4",
]

QUESTION_TEMPLATE = (
    "You are a Sudoku expert. Think the puzzle step by step, but only return the answer matrix. Always:\n"
    "1. Focus on the empty cell you are currently solving (e.g., 'E5').\n"
    "2. Check its constraints:\n"
    "- Row: '[existing numbers]'\n"
    "- Column: '[existing numbers]'\n"
    "- Palace: '[existing numbers]'\n"
    "- List possible numbers: '[Candidate numbers]'\n" 
    "3. Only place a number under the following conditions:\n"  
    "- It is the only candidate number ('Place [number] in [position] (unique candidate)'), or it is the only possible position in its row/column/region ('Place [number] in [position] (row/column/region unique)').\n" 
    "4. After placing the number, rescan the affected area.\n"
    "5. You can use the following approach: 'Hmm... A1 could be 3 or 5', 'Check box 2...', 'Verify: no conflicts?'\n\n"
    "Now you will see an image of an Sudoku puzzle. Each image is a complete Sudoku puzzle, with sizes possibly being 4x4, 6x6, or 9x9. A 6x6 palace consists of 2 rows and 3 columns. You need to understand the constraints based on all the numbers in the diagram and solve the Sudoku completely."
    "Please read this {N}x{N} puzzle and provide the complete Sudoku solution matrix, returning it in standard nested list format: [[...], [...], ...]."
)

def _mapping_json_to_list(json_obj: dict) -> List[dict]:
    """Convert mapping-style json (key -> dict) to list-of-dicts with 'puzzle_path' preserved."""
    out = []
    for k, v in json_obj.items():
        if isinstance(v, dict):
            entry = dict(v)
            # original key is something like "4x4/puzzle/puzzle_004_00000.png" — keep it
            entry.setdefault("puzzle_key", k)
            # many entries already include puzzle_path; keep or add it
            if "puzzle_path" not in entry:
                entry["puzzle_path"] = k
            out.append(entry)
    return out

def load_raw_dataset(script_args):
    """Load dataset from JSON (mapping or list) or from hub."""
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        with open(script_args.dataset_name, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data_list = _mapping_json_to_list(raw)
            ds = HFDataset.from_list(data_list)
            return DatasetDict({"train": ds})
        elif isinstance(raw, list):
            ds = HFDataset.from_list(raw)
            return DatasetDict({"train": ds})
        else:
            raise ValueError("Unsupported JSON schema for dataset_name.")
    else:
        return load_dataset(script_args.dataset_name, name=getattr(script_args, "dataset_config", None))


def grid_to_string(grid: List[List[int]]) -> str:
    """Serialize a 2D grid to newline-separated rows (space separated numbers)."""
    return "\n".join(" ".join(str(int(x)) for x in row) for row in grid)


def make_puzzle_conversation(example: dict):
    """
    Convert a puzzle example to the conversation format expected by trainer.
    We set:
      - example['problem'] : instruction text asking model to output final solution grid
      - example['problem_type'] : ['puzzle']  (reward reads kwargs['problem_type'][0])
      - example['solution'] : a string containing <answer>...</answer> with the ground-truth grid
      - example['data_type'] : 'image' (and we include puzzle_path in the prompt content)
    NOTE: adjust the content structure if your trainer expects a different image representation.
    """
    N = example.get("N", None)
    puzzle_grid = example.get("puzzle", None)
    solution_grid = example.get("solution", None)
    solution_str = "<answer>" + json.dumps(solution_grid, ensure_ascii=False) + "</answer>"
    puzzle_path = example.get("puzzle_path", example.get("puzzle_key", None))

    # Construct a human-readable problem prompt
    prob_text = QUESTION_TEMPLATE.format(N=N)

    # solution string for reward: put ground-truth inside <answer> tags

    # Make conversation prompt structure similar to your original make_conversation_image_and_video
    msg = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    # Here we include both type and path; adapt if your trainer expects only 'type' or different key
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT + "\n" + prob_text},
                ],
            }
        ],
        # keep fields used by reward and curricula
        "problem": prob_text,
        # reward expects solution param later; we store the string form
        "solution": solution_str,
        # reward expects problem_type in kwargs; earlier code used kwargs['problem_type'][0]
        "problem_type": "puzzle",
        "data_type": "image",
        "puzzle_path": puzzle_path,
        "difficulty": example.get("difficulty", "A1"),  # ensure difficulty present
    }

    return msg

# --------------------------- Curriculum controller & evaluation ---------------------------

def split_by_difficulty(full_ds: HFDataset, difficulty_field: str = "difficulty"):
    groups = {lab: [] for lab in DEFAULT_LABEL_ORDER}
    other = []
    for ex in full_ds:
        lab = ex.get(difficulty_field)
        if lab in groups:
            groups[lab].append(ex)
        else:
            other.append(ex)
    out = {}
    for lab in DEFAULT_LABEL_ORDER:
        if len(groups[lab]) == 0:
            out[lab] = HFDataset.from_dict({k: [] for k in full_ds.column_names})
        else:
            out[lab] = HFDataset.from_list(groups[lab])
    if len(other) > 0:
        out["_OTHER_"] = HFDataset.from_list(other)
    return out

def sample_dataset(ds: HFDataset, ratio: float, seed: int = 42):
    if len(ds) == 0:
        return ds
    k = max(1, int(math.ceil(len(ds) * ratio)))
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    return ds.select(idxs[:k])

def evaluate_subset_accuracy_with_model(self, trainer_cls, sample_ds: HFDataset):
    """
    Instantiate a temporary trainer with eval_dataset=sample_ds and call evaluate().
    Return a float accuracy estimate.
    If evaluate() fails or no accuracy-like field is present, returns 0.0 and prints diagnostics.
    """

    if len(sample_ds) == 0:
        return 0.0
    try:
        self.trainer.eval_dataset = sample_ds
        metrics = self.trainer.evaluate()

        # find accuracy-like key
        for k, v in metrics.items():
            if "acc" in k.lower() or "accuracy" in k.lower():
                try:
                    return float(v)
                except:
                    continue
        for k in ("eval_reward", "eval_score", "reward", "eval_loss"):
            if k in metrics:
                try:
                    return float(metrics[k])
                except:
                    continue
        numeric_vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
        if numeric_vals:
            return sum(numeric_vals) / len(numeric_vals)
        print("[Curriculum] evaluate() returned metrics but no accuracy-like field:", metrics)
        return 0.0
    except Exception as e:
        print(f"[Curriculum] evaluate() failed: {e}")
        return 0.0

def find_latest_checkpoint(class_dir):
    if not os.path.exists(class_dir):
        return None

    checkpoint_pattern = re.compile(r'checkpoint-(\d+)$')
    checkpoints = []
    
    for item in os.listdir(class_dir):
        item_path = os.path.join(class_dir, item)
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item_path))
    
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    return latest_path

@dataclass
class SubsetState:
    label: str
    dataset: HFDataset
    done: bool = False
    last_accuracy: float = 0.0

class CurriculumController:
    def __init__(self, full_ds: HFDataset, trainer_cls, script_args, training_args, model_args, initial_checkpoint: str, output_dir: str):
        self.trainer_cls = trainer_cls
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.output_dir = output_dir

        self.subsets = split_by_difficulty(full_ds, difficulty_field="difficulty")
        self.order = [lab for lab in DEFAULT_LABEL_ORDER if lab in self.subsets]
        self.states: Dict[str, SubsetState] = {lab: SubsetState(label=lab, dataset=self.subsets[lab]) for lab in self.order}

        # checkpoint tracking
        self.class_checkpoint: Dict[str, Optional[str]] = {"A": None, "B": None, "C": None}
        self.current_checkpoint = initial_checkpoint

        self.current_label: Optional[str] = None
        self.same_counter = 0
        self.history = []
        self.log_acc_path = os.path.join(self.output_dir, "accuracy_test.json")

        # *** CRITICAL CHANGE: single trainer for entire curriculum ***
        # Create the trainer ONCE during __init__.  We will reuse it throughout
        # the curriculum by swapping train_dataset and eval_dataset, avoiding
        # the DeepSpeed ZeRO-3 global-state conflicts that caused OOM.
        self.trainer = None

    def _ensure_trainer(self, method):
        """
        Lazy-initialize the single trainer instance.

        Called before the first training or evaluation.  After this, the same
        trainer is reused for all subsets by swapping its train_dataset and
        eval_dataset attributes.  This avoids recreating DeepSpeed engines,
        which cannot be properly cleaned up in ZeRO-3 due to global NCCL state.
        """
        if self.trainer is not None:
            return

        # Use a dummy dataset for initialization (will be replaced before use)
        dummy_ds = self.subsets[self.order[0]].select(range(min(2, len(self.subsets[self.order[0]]))))

        if method == "train":
            self.trainer = self.trainer_cls(
                model=self.current_checkpoint,
                reward_funcs=[reward_funcs_registry[fn] for fn in self.script_args.reward_funcs],
                args=self.training_args,
                script_args=self.script_args,
                train_dataset=dummy_ds,
                eval_dataset=None,
                peft_config=get_peft_config(self.model_args),
                attn_implementation=self.model_args.attn_implementation,
                max_pixels=self.script_args.max_pixels,
                min_pixels=self.script_args.min_pixels,
            )
            print(f"[Curriculum] Created single trainer instance (will be reused for all subsets)")
        else:
            self.trainer = self.trainer_cls(
                model=self.current_checkpoint,
                reward_funcs=[accuracy_reward],
                args=self.training_args,
                script_args=self.script_args,
                train_dataset=dummy_ds,
                eval_dataset=None,
                peft_config=get_peft_config(self.model_args),
                attn_implementation=self.model_args.attn_implementation,
                max_pixels=self.script_args.max_pixels,
                min_pixels=self.script_args.min_pixels,
            )
            print(f"[Curriculum] Created eval trainer instance (will be reused for all subsets)")

        # ── Patch 1: preserve deepspeed_engine_wrapped across free_memory() calls ──
        # _inner_training_loop calls accelerator.free_memory() at its very start.
        # The original free_memory() calls engine.destroy(), which:
        #   (a) is irreversible – global NCCL state cannot be reconstructed, and
        #   (b) raises AttributeError in DeepSpeed 0.15.x when
        #       _DeepSpeedZeroOptimizer_Stage3.__ipg_bucket_flat_buffer has not
        #       been initialised (happens when destroy() is called before the
        #       first optimizer step, e.g. on the second curriculum subset).
        # Fix: skip engine.destroy() entirely and only perform the non-destructive
        # parts of free_memory() – gc, CUDA cache flush, and bookkeeping list reset.
        # This is all that _inner_training_loop actually needs from free_memory().
        _accel = self.trainer.accelerator

        def _curriculum_free_memory(*args, **kwargs):
            # Do NOT call the original free_memory(); it destroys the ZeRO-3 engine.
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Reset accelerator bookkeeping lists so the next training loop
            # initialisation starts from a clean state.
            for _attr in ("_models", "_optimizers", "_schedulers"):
                _lst = getattr(_accel, _attr, None)
                if isinstance(_lst, list):
                    _lst.clear()

        _accel.free_memory = _curriculum_free_memory

        # ── Patch 2: reuse DeepSpeedOptimizerWrapper across deepspeed_init() calls ──
        # deepspeed_init() is called unconditionally at the start of every
        # _inner_training_loop.  It resets trainer.optimizer to None and creates a
        # brand-new plain PyTorch optimizer, which is NOT connected to the existing
        # ZeRO-3 engine and cannot update sharded parameters correctly.
        # We detect whether a DeepSpeedOptimizerWrapper is already in place; if so
        # we skip reinitialisation and return the existing optimizer/scheduler so
        # the ZeRO-3 engine remains the sole owner of parameter updates.
        import transformers.trainer as _trainer_module
        from accelerate.utils import DeepSpeedOptimizerWrapper as _DSOptWrapper
        _orig_deepspeed_init = _trainer_module.deepspeed_init

        def _curriculum_deepspeed_init(trainer, num_training_steps, **kwargs):
            if isinstance(getattr(trainer, 'optimizer', None), _DSOptWrapper):
                # Engine already initialised for this trainer – reuse it.
                return trainer.optimizer, trainer.lr_scheduler
            return _orig_deepspeed_init(trainer, num_training_steps, **kwargs)

        _trainer_module.deepspeed_init = _curriculum_deepspeed_init

    def _log_eval_results(self, res: dict):
        """Append accuracy results to the JSON log file."""
        log_entry = {'timestamp': datetime.now().isoformat(), 'accuracies': res}
        if os.path.exists(self.log_acc_path):
            with open(self.log_acc_path, 'r') as f:
                try:
                    log_data = json.load(f)
                    if not isinstance(log_data, list):
                        log_data = [log_data]
                except json.JSONDecodeError:
                    log_data = []
        else:
            log_data = []
        log_data.append(log_entry)
        with open(self.log_acc_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _run_evaluate(self) -> dict:
        """
        Evaluate all undone subsets using the single trainer instance.

        Temporarily swaps trainer.eval_dataset, runs evaluation, then restores
        the original dataset.  Returns {label -> accuracy}.
        """
        self._ensure_trainer("eval")
        FIXED_EVAL_SIZE = self.script_args.eval_fixed_size
        res = {}

        from datasets import concatenate_datasets
        all_samples = []
        for lab in self.order:
            st = self.states[lab]
            if st.done or len(st.dataset) == 0:
                st.last_accuracy = 0.0
                res[lab] = 0.0
                continue
            sample = sample_dataset(st.dataset, self.script_args.eval_sample_ratio, seed=0)
            if len(sample) > FIXED_EVAL_SIZE:
                sample = sample.select(range(FIXED_EVAL_SIZE))
            all_samples.append(sample)

        if not all_samples:
            return res

        old_eval_ds = self.trainer.eval_dataset
        old_train_mode = self.trainer.model.training
        try:
            self.trainer.eval_dataset = concatenate_datasets(all_samples)
            self.trainer.model.eval()

            with torch.inference_mode():
                metrics = self.trainer.evaluate()

            eval_loss_list = metrics.get('eval_loss_list', [])
            eval_labels   = metrics.get('eval_labels', [])

            label_metrics = {}
            for label_num in set(eval_labels):
                indices = [i for i, l in enumerate(eval_labels) if l == label_num]
                if indices:
                    label_metrics[label_num] = sum(eval_loss_list[i] for i in indices) / len(indices)

            for lab in self.order:
                st = self.states[lab]
                if st.done or len(st.dataset) == 0:
                    continue
                acc = label_metrics.get(difficulty_mapping[lab], 0.0)
                st.last_accuracy = acc
                res[lab] = acc
                print(f"[Curriculum] eval {lab}: acc={acc:.4f} sample_size={FIXED_EVAL_SIZE}")

        finally:
            self.trainer.eval_dataset = old_eval_ds
            if old_train_mode:
                self.trainer.model.train()

        return res

    def curriculum_eval(self):
        """
        Evaluate all undone subsets and return {label -> accuracy}.

        Uses the single trainer instance (created on first call via
        _ensure_trainer).  Temporarily swaps eval_dataset, runs evaluation,
        then restores the original state.
        """
        res = self._run_evaluate()
        self._log_eval_results(res)
        return res

    def pick_best_undone(self):
        undone = {lab: st.last_accuracy for lab, st in self.states.items() if not st.done and len(st.dataset) > 0}
        if not undone:
            return None
        return max(undone.items(), key=lambda kv: kv[1])[0]

    def _class_of_label(self, label: str) -> str:
        # label like "A1" -> class "A"
        return label[0] if label and len(label) > 0 else "A"

    def train_subset(self, label: str, epochs: Optional[int] = None):
        """
        Train on *label*'s dataset using the single trainer instance.

        Swaps trainer.train_dataset to the target subset, updates reward_funcs
        if needed, then calls trainer.train().  The same trainer is reused
        across all subsets, avoiding DeepSpeed ZeRO-3 reinitialization issues.

        Args:
            label:  subset key to train on.
            epochs: number of training epochs; defaults to script_args.epochs_per_subset.
        """
        self._ensure_trainer("train")
        st = self.states[label]
        class_letter = self._class_of_label(label)
        effective_epochs = epochs if epochs is not None else self.script_args.epochs_per_subset
        print(f"[Curriculum] === Training on subset {label} (class {class_letter}) "
              f"size={len(st.dataset)} epochs={effective_epochs} ===")

        # Update output directory for this class
        self.training_args.output_dir = os.path.join(self.output_dir, f"model_{class_letter}")
        os.makedirs(self.training_args.output_dir, exist_ok=True)

        # Swap the training dataset
        self.trainer.train_dataset = st.dataset

        # Reset trainer state for new training run
        self.trainer.args.num_train_epochs = effective_epochs
        self.trainer.state.global_step = 0
        self.trainer.state.epoch = 0
        self.trainer.model.train()

        # Reset LR scheduler so each subset gets a fresh cosine curve sized to
        # its own step count (avoids training at near-zero LR after the first subset).
        # Skip on the first call: optimizer is None until DeepSpeed initializes inside train().
        if self.trainer.optimizer is not None:
            num_steps = math.ceil(
                len(st.dataset) / (
                    self.training_args.per_device_train_batch_size
                    * self.training_args.world_size
                    * self.training_args.gradient_accumulation_steps
                )
            ) * effective_epochs
            self.trainer.create_scheduler(num_training_steps=num_steps, optimizer=self.trainer.optimizer)
            print(f"[Curriculum] Reset scheduler for '{label}': num_training_steps={num_steps}")

        # Train
        self.trainer.train()

        # Save checkpoint
        last_checkpoint = find_latest_checkpoint(self.training_args.output_dir)
        if last_checkpoint is not None:
            self.current_checkpoint = last_checkpoint
        else:
            self.trainer.save_model(self.training_args.output_dir)
            self.current_checkpoint = self.training_args.output_dir
        self.class_checkpoint[class_letter] = self.current_checkpoint
        print(f"[Curriculum] Saved model for class {class_letter} at {self.current_checkpoint}")

    def run(self):
        """
        Main curriculum loop using a single trainer instance.

        Architecture change
        ───────────────────
        The old multi-trainer approach (create → train → destroy → create next)
        failed because DeepSpeed ZeRO-3 maintains process-global state (NCCL
        communicators, parameter partition tables) that cannot be reset within
        a single process.  Calling engine.destroy() only frees optimizer state
        and gradient buffers, not the global infrastructure.

        New approach: ONE trainer for the entire curriculum.  We swap its
        train_dataset and eval_dataset attributes between subsets.  This avoids
        recreating DeepSpeed engines and eliminates the OOM issue.
        """
        # ── Pick first subset without pre-evaluation ──────────────────────
        # Skipping initial curriculum_eval() intentionally: calling
        # trainer.evaluate() before trainer.train() causes DeepSpeed to
        # initialise only in inference mode, leaving deepspeed_engine_wrapped
        # as None and crashing on the first backward() call.
        # All last_accuracy values start at 0.0 anyway so the initial eval
        # would not change which subset gets picked first.

        # If start_subset is specified, mark all earlier subsets as done.
        requested_start = self.script_args.start_subset
        if requested_start is not None:
            if requested_start not in difficulty_mapping:
                raise ValueError(
                    f"--start_subset '{requested_start}' is not a valid subset label. "
                    f"Valid labels: {list(difficulty_mapping.keys())}"
                )
            start_idx = difficulty_mapping[requested_start]
            for lab in self.order:
                if difficulty_mapping[lab] < start_idx:
                    self.states[lab].done = True
                    print(f"[Curriculum] Skipping subset '{lab}' (start_subset={requested_start})")

        start = next(
            (lab for lab in self.order
             if not self.states[lab].done and len(self.states[lab].dataset) > 0),
            None,
        )
        if start is None:
            print("[Curriculum] No data in any subset. Exiting.")
            return
        self.current_label = start
        self.same_counter = 0
        next_epochs = self.script_args.epochs_per_subset  # may be doubled by threshold_for_double

        # ── Main loop ─────────────────────────────────────────────────────
        while True:
            # 1. Train (reuses self.trainer, swaps train_dataset)
            t0 = time.time()
            self.train_subset(self.current_label, epochs=next_epochs)
            print(f"[Curriculum] Training time: {time.time()-t0:.1f}s")

            self.states[self.current_label].done = True
            self.history.append(self.current_label)

            undone_exist = any(
                not st.done and len(st.dataset) > 0 for st in self.states.values()
            )
            if not undone_exist:
                print("[Curriculum] All subsets done. Exiting.")
                break

            # 2. Evaluate (reuses self.trainer, swaps eval_dataset)
            t0 = time.time()
            accs = self.curriculum_eval()
            print(f"[Curriculum] Eval time: {time.time()-t0:.1f}s")

            # 3. Pick next subset
            threshold_low  = self.script_args.accuracy_threshold_for_switch
            threshold_high = self.script_args.accuracy_threshold_upper
            strategy       = self.script_args.curriculum_selection_strategy
            candidates = {
                lab: st.last_accuracy
                for lab, st in self.states.items()
                if not st.done and threshold_low < st.last_accuracy <= threshold_high
            }
            if not candidates:
                next_label = self.current_label
            elif strategy == "median":
                sorted_cands = sorted(candidates.items(), key=lambda kv: kv[1])
                next_label = sorted_cands[len(sorted_cands) // 2][0]
            elif strategy == "random":
                next_label = random.choice(list(candidates.keys()))
            else:  # argmax (default)
                next_label = max(candidates.items(), key=lambda kv: kv[1])[0]
            print(f"[Curriculum] strategy={strategy} candidates={candidates} → next={next_label}")

            if next_label == self.current_label:
                self.same_counter += 1
            else:
                self.same_counter = 0

            if self.same_counter >= self.script_args.early_stop_same_repeat:
                print(
                    f"[Curriculum] Subset '{self.current_label}' trained "
                    f"{self.same_counter} consecutive times – early stopping."
                )
                break

            self.current_label = next_label

            # Compute epochs for the next training run.
            # If accuracy_threshold_for_double is set and the chosen subset's
            # accuracy is still below it, double the epoch count to give the
            # model more steps on the harder material.
            next_epochs = self.script_args.epochs_per_subset
            td = self.script_args.accuracy_threshold_for_double
            if td is not None:
                next_acc = self.states[self.current_label].last_accuracy
                if next_acc < td:
                    next_epochs = self.script_args.epochs_per_subset * 2
                    print(
                        f"[Curriculum] Doubling epochs for '{self.current_label}' "
                        f"(acc={next_acc:.4f} < threshold_for_double={td}) "
                        f"→ epochs={next_epochs}"
                    )

        print("[Curriculum] Curriculum finished. History:", self.history)

# --------------------------- main() ---------------------------

def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load dataset (supports mapping-style puzzle.json)
    dsdict = load_raw_dataset(script_args)
    # get train split
    if "train" in dsdict:
        full_ds = dsdict["train"]
    else:
        # if load_dataset returned dataset dict with other split names
        first_split = list(dsdict.values())[0]
        full_ds = first_split

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("[Main] Using trainer class:", trainer_cls)

    if script_args.use_curriculum:
        print("[Main] Curriculum mode enabled.")
        # Convert all examples to conversation format (make_puzzle_conversation)
        # Note: map can be heavy on memory; do in place mapping
        conv_ds = full_ds.map(make_puzzle_conversation, remove_columns=full_ds.column_names)
        # conv_ds now has fields like 'prompt','solution','problem_type','difficulty',...
        # Build controller and run
        controller = CurriculumController(conv_ds, trainer_cls, script_args, training_args, model_args, model_args.model_name_or_path, training_args.output_dir)
        controller.run()
        print("[Main] Curriculum run complete.")
        return

    # Fallback: normal single-dataset training (no curriculum)
    dataset = full_ds.map(make_puzzle_conversation, remove_columns=full_ds.column_names)
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
