"""Rebuild missing/invalid Sudoku images and traces from numeric metadata."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.data import REPO_ROOT, load_puzzles
from open_r1.sudoku import count_solutions, replay_trace, solve_with_trace, trace_cost


def asset_output(reference):
    root = (REPO_ROOT / "data").resolve()
    target = (root / reference).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Asset path escapes data directory: {reference}")
    return target


def render_grid(grid, reference_image, reference_grid, output):
    """Reuse the released test rendering's grid and digit glyphs, without resizing."""
    from PIL import Image, ImageDraw, PngImagePlugin

    size, margin, cell_width = len(grid), 20, 50
    with Image.open(reference_image) as source:
        image = source.convert("RGBA")
        if image.size != (size * cell_width + 2 * margin,) * 2:
            raise ValueError(f"Unexpected style reference size: {source.size}")
        glyphs = {}
        for r, row in enumerate(reference_grid):
            for c, value in enumerate(row):
                left, top = margin + c * cell_width, margin + r * cell_width
                glyphs.setdefault(value, image.crop((left + 3, top + 3, left + 48, top + 48)))
        if set(glyphs) != set(range(1, size + 1)):
            raise ValueError("Incomplete digit style reference")
        draw = ImageDraw.Draw(image)
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                left, top = margin + c * cell_width, margin + r * cell_width
                draw.rectangle((left + 3, top + 3, left + 47, top + 47), fill="white")
                if value:
                    image.paste(glyphs[value], (left + 3, top + 3))
        info = PngImagePlugin.PngInfo()
        info.add_text("generator", "fg-sudoku-release-v1")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, pnginfo=info)


def valid_image(path):
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def restore(dataset):
    counts = Counter()
    references = load_puzzles(REPO_ROOT / "data/sudoku_test/puzzles.json")
    styles = {size: next(item for item in references.values() if item["N"] == size) for size in (4, 6, 9)}
    for key, item in load_puzzles(dataset).items():
        # Solve independently: reference solutions are used only for verification.
        solved, trace = solve_with_trace(item["puzzle"])
        if solved != item["solution"] or trace_cost(trace) != item["steps"]:
            raise ValueError(f"{key}: regenerated solution/cost differs from metadata")
        if count_solutions(item["puzzle"]) != 1:
            raise ValueError(f"{key}: puzzle is not uniquely solvable")
        trace_path = asset_output(item["trace_path"])
        try:
            existing = json.loads(trace_path.read_text(encoding="utf-8"))
            if replay_trace(item["puzzle"], existing) != solved or trace_cost(existing) != item["steps"]:
                raise ValueError("Trace mismatch")
        except (OSError, ValueError, KeyError, TypeError):
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            counts["traces"] += 1
        style = styles[item["N"]]
        for field, grid in (("puzzle_path", item["puzzle"]), ("solution_path", solved)):
            output = asset_output(item[field])
            if not valid_image(output):
                render_grid(grid, REPO_ROOT / "data" / style["solution_path"], style["solution"], output)
                counts[field] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/sudoku/puzzles.json")
    args = parser.parse_args()
    print(json.dumps(restore(args.dataset), indent=2))


if __name__ == "__main__":
    main()
