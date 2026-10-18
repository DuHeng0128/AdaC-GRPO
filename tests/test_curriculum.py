import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_r1.curriculum import evaluation_sample_size, select_group, restart_deepspeed_scheduler


class CurriculumTests(unittest.TestCase):
    def test_inclusive_zpd_and_argmax(self):
        self.assertEqual(select_group({"A1": 0.3, "A2": 0.1}, "A2"), "A1")
        self.assertEqual(select_group({"A1": 0.8, "A2": 0.5}, "A2"), "A1")
        self.assertEqual(select_group({"A1": 0.81, "A2": 0.6}, "A1"), "A2")

    def test_fallback_and_mastery(self):
        self.assertEqual(select_group({"A1": 0.2, "A2": 0.1}, "A1"), "A1")
        self.assertIsNone(select_group({"A1": 0.9, "A2": 0.95}, "A1"))
        self.assertIsNone(select_group({}, "A1"))
        with self.assertRaises(ValueError):
            select_group({"A1": 0.5}, "A1", strategy="invalid")

    def test_online_ratio_without_hidden_cap(self):
        self.assertEqual(evaluation_sample_size(376), 94)
        self.assertEqual(evaluation_sample_size(271), 68)
        self.assertEqual(evaluation_sample_size(376, cap=16), 16)
        self.assertEqual(evaluation_sample_size(0), 0)
        with self.assertRaises(ValueError):
            evaluation_sample_size(10, ratio=0)

    def test_controller_consolidates_until_mastered(self):
        # Exercise the real loop with lightweight trainer/evaluation stand-ins.
        tree = ast.parse((ROOT / "src/open_r1/grpo.py").read_text(encoding="utf-8"))
        controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CurriculumController")
        run = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        scope = {"time": SimpleNamespace(time=lambda: 0), "difficulty_mapping": {"A1": 0, "B1": 4}, "select_group": select_group}
        exec(compile(ast.Module(body=[run], type_ignores=[]), "curriculum-test", "exec"), scope)
        rounds = iter([{"A1": 0.5, "B1": 0.4}, {"A1": 0.9, "B1": 0.6}, {"B1": 0.9}])
        fake = SimpleNamespace(
            script_args=SimpleNamespace(start_subset=None, epochs_per_subset=3,
                accuracy_threshold_for_switch=0.3, accuracy_threshold_upper=0.8,
                curriculum_selection_strategy="argmax", early_stop_same_repeat=3,
                accuracy_threshold_for_double=None),
            states={label: SimpleNamespace(done=False, last_accuracy=0, dataset=[1]) for label in ("A1", "B1")},
            order=["A1", "B1"], history=[], train_subset=lambda *args, **kwargs: None,
        )

        def evaluate():
            accuracies = next(rounds)
            for label, accuracy in accuracies.items():
                fake.states[label].last_accuracy = accuracy
            return accuracies

        fake.curriculum_eval = evaluate
        with contextlib.redirect_stdout(io.StringIO()):
            scope["run"](fake)
        self.assertEqual(fake.history, ["A1", "A1", "B1"])
        self.assertTrue(all(state.done for state in fake.states.values()))

    def test_scheduler_restart_preserves_engine_optimizer(self):
        raw_optimizer, wrapped_optimizer = object(), object()
        engine = SimpleNamespace(lr_scheduler=SimpleNamespace(optimizer=raw_optimizer))
        trainer = SimpleNamespace(
            optimizer=wrapped_optimizer, lr_scheduler=None,
            accelerator=SimpleNamespace(deepspeed_engine_wrapped=SimpleNamespace(engine=engine)),
        )

        def create_scheduler(num_training_steps, optimizer):
            self.assertIsNone(trainer.lr_scheduler)
            self.assertIs(optimizer, raw_optimizer)
            trainer.lr_scheduler = SimpleNamespace(optimizer=optimizer, steps=num_training_steps)

        trainer.create_scheduler = create_scheduler
        wrapper = lambda scheduler, optimizer: SimpleNamespace(scheduler=scheduler, optimizer=optimizer)
        optimizer, scheduler = restart_deepspeed_scheduler(trainer, 123, wrapper)
        self.assertIs(optimizer, wrapped_optimizer)
        self.assertIs(scheduler.scheduler, engine.lr_scheduler)
        self.assertEqual(engine.lr_scheduler.steps, 123)
        self.assertTrue(scheduler._is_accelerate_prepared)
        restart_deepspeed_scheduler(trainer, 45, wrapper)
        self.assertEqual(engine.lr_scheduler.steps, 45)


if __name__ == "__main__":
    unittest.main()
