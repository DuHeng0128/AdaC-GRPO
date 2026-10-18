import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_r1.data import load_puzzles, resolve_asset
from open_r1.sudoku import count_solutions, replay_trace, solve_with_trace, trace_cost


class AssetTests(unittest.TestCase):
    def test_recomputed_solutions_and_costs_match_every_puzzle(self):
        for split in ("sudoku", "sudoku_test"):
            for key, item in load_puzzles(ROOT / "data" / split / "puzzles.json").items():
                solution, trace = solve_with_trace(item["puzzle"])
                self.assertEqual(solution, item["solution"], key)
                self.assertEqual(trace_cost(trace), item["steps"], key)

    def test_replay_rejects_corrupted_trace(self):
        item = next(iter(load_puzzles(ROOT / "data/sudoku/puzzles.json").values()))
        solution, trace = solve_with_trace(item["puzzle"])
        self.assertEqual(replay_trace(item["puzzle"], trace), solution)
        broken = copy.deepcopy(trace)
        broken[0]["candidates_before"] = []
        with self.assertRaises(ValueError):
            replay_trace(item["puzzle"], broken)
        self.assertEqual(count_solutions(item["puzzle"]), 1)

    def test_sft_images_are_nonempty(self):
        path = ROOT / "data/sudoku/sft.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 300)
        for row in rows:
            self.assertGreater(Path(resolve_asset(row["path"], path)).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
