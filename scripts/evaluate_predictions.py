"""Score saved Sudoku predictions with strict exact-grid matching (CPU only)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.data import LABELS, REPO_ROOT, extract_matrix, load_puzzles


def score_predictions(puzzles, predictions, allow_partial=False):
    unknown = set(predictions) - set(puzzles)
    if unknown:
        raise ValueError(f"Unknown prediction keys: {sorted(unknown)[:3]}")
    if not allow_partial and set(predictions) != set(puzzles):
        raise ValueError(f"Expected {len(puzzles)} predictions, found {len(predictions)}; --allow-partial scores only supplied keys")
    keys = list(predictions) if allow_partial else list(puzzles)
    if not keys:
        raise ValueError("No predictions to evaluate")
    counts = {label: {"correct": 0, "total": 0} for label in LABELS}
    invalid = 0
    for key in keys:
        item = puzzles[key]
        predicted = extract_matrix(predictions[key], item["N"])
        invalid += predicted is None
        bucket = counts[item["difficulty"]]
        bucket["total"] += 1
        bucket["correct"] += int(predicted == item["solution"])
    for group in "ABC":
        counts[group] = {field: sum(counts[f"{group}{tier}"][field] for tier in range(1, 5)) for field in ("correct", "total")}
    counts["overall"] = {field: sum(counts[group][field] for group in "ABC") for field in ("correct", "total")}
    for bucket in counts.values():
        bucket["pass_at_1"] = bucket["correct"] / bucket["total"] if bucket["total"] else None
    return {"partial": allow_partial, "invalid_predictions": invalid, "results": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON mapping puzzles.json image keys to matrices or response text")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/sudoku_test/puzzles.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new path")
    report = score_predictions(load_puzzles(args.dataset), json.loads(args.predictions.read_text(encoding="utf-8")), args.allow_partial)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
