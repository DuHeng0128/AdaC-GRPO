# FG-Sudoku

`data/sudoku/puzzles.json` contains 3,072 training puzzles;
`data/sudoku_test/puzzles.json` contains 768 held-out puzzles.
Each split has equal numbers of 4×4, 6×6, and 9×9 grids, divided into four
solver-cost tiers per size. The 6×6 block shape is 2×3.

| Split / grid | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
| --- | ---: | ---: | ---: | ---: |
| Train 4×4 | 376 | 271 | 239 | 138 |
| Train 6×6 | 324 | 245 | 234 | 221 |
| Train 9×9 | 308 | 246 | 214 | 256 |
| Test 4×4 | 85 | 97 | 53 | 21 |
| Test 6×6 | 82 | 64 | 62 | 48 |
| Test 9×9 | 69 | 59 | 68 | 60 |

## Metadata and assets

Each image key maps to `N`, `puzzle` (zeros denote blanks), `solution`,
`difficulty`, `steps`, `steps_log1p`, `drop_ratio`, and three asset paths:
`puzzle_path`, `solution_path`, `trace_path`. Paths resolve relative to the
dataset or repository data directory. `data/sudoku/sft.json` provides the
300 warm-start demonstrations.

Solver traces record assignments and candidate eliminations. Their weighted
cost is `2 * assignments + eliminations`, matching `steps`. Recomputed training
traces identify their generator in the first event; regenerated images retain
the test split's grid and digit style. Numeric puzzles, answers, split
membership, and difficulty labels are unchanged.

```bash
# Check all PNGs, replay every trace, and compare answers and costs.
python scripts/validate_data.py

# Also verify unique solvability by independent MRV search.
python scripts/validate_data.py --check-uniqueness

# Rebuild missing or invalid assets when needed (requires Pillow).
python -m pip install -e '.[data]'
python scripts/restore_assets.py
```

## Predictions

Save a JSON object mapping the original image keys to matrices or response
strings. Strings may contain an `<answer>...</answer>` block.

```json
{
  "4x4/puzzle/puzzle_004_00000.png": [[1, 3, 4, 2], [2, 4, 1, 3], [3, 1, 2, 4], [4, 2, 3, 1]]
}
```

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/test_predictions.json \
  --output outputs/test_metrics.json
```

Full evaluation requires all 768 keys. Unknown keys are rejected; malformed
or incorrect grids count as failures. `--allow-partial` explicitly permits a
smaller set and marks its report as partial. `examples/predictions.json` is a
reference-answer format example, not a model prediction.

The evaluator reports exact-grid pass@1 per tier, per size, and overall,
weighted by example counts. Auxiliary training rewards are not benchmark scores.
