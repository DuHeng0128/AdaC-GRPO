"""CPU-only curriculum selection and sampling rules."""

import math
import random


def evaluation_sample_size(count, ratio=0.25, cap=0):
    if count < 0 or not 0 < ratio <= 1 or cap < 0:
        raise ValueError("Expected count/cap >= 0 and 0 < ratio <= 1")
    size = min(count, max(1, math.ceil(count * ratio))) if count else 0
    return min(size, cap) if cap else size


def select_group(accuracies, previous, low=0.3, high=0.8, strategy="argmax"):
    if not 0 <= low <= high <= 1:
        raise ValueError("Expected 0 <= low <= high <= 1")
    if strategy not in ("argmax", "median", "random"):
        raise ValueError(f"Unknown selection strategy: {strategy}")
    if not accuracies or all(accuracy > high for accuracy in accuracies.values()):
        return None
    candidates = {label: accuracy for label, accuracy in accuracies.items() if low <= accuracy <= high}
    if not candidates:
        return previous
    if strategy == "median":
        ordered = sorted(candidates, key=candidates.get)
        return ordered[len(ordered) // 2]
    if strategy == "random":
        return random.choice(list(candidates))
    return max(candidates, key=candidates.get)


def restart_deepspeed_scheduler(trainer, num_training_steps, scheduler_wrapper):
    """Restart the subset schedule without replacing the ZeRO optimizer/engine.

    DeepSpeed, not Trainer, owns scheduler stepping. Both must refer to the
    new scheduler, with a no-op Accelerate wrapper to avoid double stepping.
    """
    engine = trainer.accelerator.deepspeed_engine_wrapped.engine
    previous = engine.lr_scheduler
    optimizer = getattr(previous, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("Curriculum scheduling requires a PyTorch scheduler with an optimizer")
    trainer.lr_scheduler = None
    trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
    engine.lr_scheduler = trainer.lr_scheduler
    trainer.lr_scheduler = scheduler_wrapper(engine.lr_scheduler, trainer.optimizer)
    trainer.lr_scheduler._is_accelerate_prepared = True
    return trainer.optimizer, trainer.lr_scheduler
