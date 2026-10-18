"""Validate the shipped FG-Sudoku files without loading a model."""

import argparse
import json
import sys
import struct
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.data import LABELS, REPO_ROOT, load_puzzles, resolve_asset, valid_solution
from open_r1.sudoku import count_solutions, replay_trace, trace_cost


def validate_png(path, size):
    """Check PNG chunks, CRCs, dimensions and compressed pixels using stdlib."""
    payload = Path(path).read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG header: {path}")
    offset, pixels, ended = 8, bytearray(), False
    width = height = bit_depth = color = None
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        tag = payload[offset + 4:offset + 8]
        content = payload[offset + 8:offset + 8 + length]
        checksum = payload[offset + 8 + length:offset + 12 + length]
        if len(content) != length or len(checksum) != 4 or zlib.crc32(tag + content) != struct.unpack(">I", checksum)[0]:
            raise ValueError(f"Invalid PNG chunk: {path}")
        if tag == b"IHDR":
            width, height, bit_depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", content)
            if (width, height) != (size * 50 + 40,) * 2 or bit_depth != 8 or color not in (2, 6) or compression or filtering or interlace:
                raise ValueError(f"Unexpected Sudoku image format: {path}")
        elif tag == b"IDAT":
            pixels.extend(content)
        elif tag == b"IEND":
            ended = True
            break
        offset += length + 12
    if not ended or width is None or not pixels:
        raise ValueError(f"Incomplete PNG: {path}")
    raw = zlib.decompress(pixels)
    expected = height * (1 + width * (4 if color == 6 else 3))
    if len(raw) != expected:
        raise ValueError(f"Invalid PNG pixel data: {path}")


def validate(dataset, uniqueness=False):
    data = load_puzzles(dataset)
    counts = Counter()
    assets = Counter()
    for key, item in data.items():
        size = item["N"]
        if size not in (4, 6, 9) or item["difficulty"] not in LABELS:
            raise ValueError(f"{key}: invalid size or difficulty")
        if item["difficulty"][0] != {4: "A", 6: "B", 9: "C"}[size]:
            raise ValueError(f"{key}: difficulty group does not match grid size")
        if not valid_solution(item["solution"], size):
            raise ValueError(f"{key}: invalid reference solution")
        puzzle = item["puzzle"]
        if len(puzzle) != size or any(len(row) != size for row in puzzle):
            raise ValueError(f"{key}: invalid puzzle dimensions")
        for r in range(size):
            for c in range(size):
                cell = puzzle[r][c]
                if type(cell) is not int or not 0 <= cell <= size or (cell and cell != item["solution"][r][c]):
                    raise ValueError(f"{key}: clue conflicts with reference solution")
        for field in ("puzzle_path", "solution_path"):
            validate_png(resolve_asset(item[field], dataset), size)
            assets[field] += 1
        trace = json.loads(Path(resolve_asset(item["trace_path"], dataset)).read_text(encoding="utf-8"))
        if replay_trace(puzzle, trace) != item["solution"] or trace_cost(trace) != item["steps"]:
            raise ValueError(f"{key}: trace solution/cost differs from metadata")
        assets["trace_path"] += 1
        if uniqueness and count_solutions(puzzle) != 1:
            raise ValueError(f"{key}: puzzle is not uniquely solvable")
        counts[item["difficulty"]] += 1
    return {"total": len(data), "tiers": {label: counts[label] for label in LABELS}, "validated_assets": dict(assets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Validate one puzzles.json; default: both shipped splits")
    parser.add_argument("--check-uniqueness", action="store_true", help="Also run independent MRV search")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else [REPO_ROOT / "data" / split / "puzzles.json" for split in ("sudoku", "sudoku_test")]
    print(json.dumps({str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): validate(path, args.check_uniqueness) for path in datasets}, indent=2))


if __name__ == "__main__":
    main()
