import json
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from open_r1.data import extract_matrix, load_puzzles, resolve_asset, valid_solution
from evaluate_predictions import score_predictions


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles = load_puzzles(ROOT / "data/sudoku_test/puzzles.json")
        cls.example = json.loads((ROOT / "examples/predictions.json").read_text())
        cls.key = next(iter(cls.example))

    def test_historical_and_dataset_relative_paths(self):
        expected = ROOT / "data/sudoku_test" / self.key
        metadata = self.puzzles[self.key]
        self.assertEqual(Path(resolve_asset(metadata["puzzle_path"], ROOT / "data/sudoku_test/puzzles.json")), expected)
        self.assertEqual(Path(resolve_asset(self.key, ROOT / "data/sudoku_test/puzzles.json")), expected)
        self.assertEqual(Path(resolve_asset(str(expected))), expected)

    def test_all_reference_solutions_are_valid(self):
        for split in ("sudoku", "sudoku_test"):
            puzzles = load_puzzles(ROOT / "data" / split / "puzzles.json")
            self.assertEqual(len(puzzles), 3072 if split == "sudoku" else 768)
            for key, item in puzzles.items():
                self.assertTrue(valid_solution(item["solution"], item["N"]), key)

    def test_prediction_parsing(self):
        matrix = self.example[self.key]
        self.assertEqual(extract_matrix(f"<think>reasoning</think><answer>{matrix}</answer>", 4), matrix)
        self.assertIsNone(extract_matrix("not a matrix", 4))
        self.assertIsNone(extract_matrix([[True] * 4] * 4, 4))
        self.assertIsNone(extract_matrix([[0] * 4] * 4, 4))
        self.assertIsNone(extract_matrix([[1, 2]], 4))

    def test_partial_evaluation_is_explicit(self):
        with self.assertRaises(ValueError):
            score_predictions(self.puzzles, self.example)
        result = score_predictions(self.puzzles, self.example, allow_partial=True)
        self.assertTrue(result["partial"])
        self.assertEqual(result["results"]["overall"]["total"], 1)
        self.assertEqual(result["results"]["overall"]["pass_at_1"], 1)

    def test_unknown_keys_and_invalid_predictions(self):
        with self.assertRaises(ValueError):
            score_predictions(self.puzzles, {"unknown.png": []}, allow_partial=True)
        result = score_predictions(self.puzzles, {self.key: "invalid"}, allow_partial=True)
        self.assertEqual(result["invalid_predictions"], 1)
        self.assertEqual(result["results"]["overall"]["pass_at_1"], 0)

    def test_overall_is_example_weighted(self):
        grid = self.example[self.key]
        puzzles = {"a": dict(N=4, difficulty="A1", solution=grid), "b": dict(N=4, difficulty="A1", solution=grid), "c": dict(N=4, difficulty="A2", solution=grid)}
        result = score_predictions(puzzles, {"a": grid, "b": grid, "c": "invalid"})
        self.assertAlmostEqual(result["results"]["A"]["pass_at_1"], 2 / 3)
        self.assertAlmostEqual(result["results"]["overall"]["pass_at_1"], 2 / 3)

    def test_sft_collation_preserves_all_images_in_batch(self):
        tree = ast.parse((ROOT / "src/open_r1/sft_video.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collate_fn")

        class FakeTensor:
            def clone(self):
                return FakeTensor()

            def __eq__(self, other):
                return None

            def __setitem__(self, key, value):
                pass

        class FakeProcessor:
            tokenizer = SimpleNamespace(pad_token_id=0)

            def apply_chat_template(self, messages, tokenize):
                return str(messages)

            def __call__(self, **kwargs):
                self.inputs = kwargs
                return {"input_ids": FakeTensor()}

        processor = FakeProcessor()
        scope = {"List": list, "Dict": dict, "Any": object,
                 "torch": SimpleNamespace(Tensor=object), "processor": processor,
                 "Qwen2VLProcessor": FakeProcessor,
                 "process_vision_info": lambda messages: ([messages], None)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "sft-test", "exec"), scope)
        scope["collate_fn"]([{"messages": "image-one"}, {"messages": "image-two"}])
        self.assertEqual(processor.inputs["images"], ["image-one", "image-two"])
        self.assertEqual(len(processor.inputs["text"]), 2)


if __name__ == "__main__":
    unittest.main()
