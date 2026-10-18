"""Dependency-free helpers for the FG-Sudoku release."""

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = tuple(f"{group}{tier}" for group in "ABC" for tier in range(1, 5))


def resolve_asset(path, dataset_file=None):
    """Resolve both historical sudoku/... paths and dataset-relative paths."""
    asset = Path(path)
    if asset.is_absolute():
        candidates = [asset]
    else:
        candidates = []
        if dataset_file:
            parent = Path(dataset_file).resolve().parent
            candidates.extend([parent / asset, parent.parent / asset])
        candidates.extend([REPO_ROOT / "data" / asset, REPO_ROOT / asset, asset])
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size:
            return str(candidate.resolve())
    raise FileNotFoundError(f"Cannot resolve dataset asset: {path}")


def load_puzzles(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected puzzles.json mapping image keys to puzzle metadata")
    return data


def extract_matrix(value, size):
    """Read an integer matrix from JSON, plain text, or an <answer> block."""
    if isinstance(value, str):
        answer = re.search(r"<answer>(.*?)</answer>", value, re.DOTALL)
        text = answer.group(1) if answer else value
        match = re.search(r"\[\s*(?:\[[^\]]+\]\s*,?\s*){%d}\]" % size, text)
        if not match:
            return None
        try:
            value = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, list) or len(value) != size:
        return None
    if any(not isinstance(row, list) or len(row) != size for row in value):
        return None
    if any(type(cell) is not int or not 1 <= cell <= size for row in value for cell in row):
        return None
    return value


def valid_solution(grid, size):
    if extract_matrix(grid, size) is None:
        return False
    expected = set(range(1, size + 1))
    if any(set(row) != expected for row in grid):
        return False
    if any({grid[r][c] for r in range(size)} != expected for c in range(size)):
        return False
    block_h, block_w = {4: (2, 2), 6: (2, 3), 9: (3, 3)}[size]
    return all(
        {grid[r][c] for r in range(top, top + block_h) for c in range(left, left + block_w)} == expected
        for top in range(0, size, block_h) for left in range(0, size, block_w)
    )
