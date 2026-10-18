"""Deterministic Sudoku solving and replay checks for dataset assets."""

from .data import valid_solution

BLOCKS = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


def units(size):
    height, width = BLOCKS[size]
    rows = [[(r, c) for c in range(size)] for r in range(size)]
    cols = [[(r, c) for r in range(size)] for c in range(size)]
    blocks = [
        [(r, c) for r in range(top, top + height) for c in range(left, left + width)]
        for top in range(0, size, height) for left in range(0, size, width)
    ]
    return rows, cols, blocks


def peers(size):
    groups = [unit for family in units(size) for unit in family]
    return {
        (r, c): sorted({cell for unit in groups if (r, c) in unit for cell in unit} - {(r, c)})
        for r in range(size) for c in range(size)
    }


def solve_with_trace(puzzle):
    """Solve by naked/hidden singles, without reading a reference answer.

    FG-Sudoku's dense clue layouts are solved by these first two rules.
    Fail explicitly if a different puzzle needs stronger deductions/search.
    """
    size = len(puzzle)
    grid = [row[:] for row in puzzle]
    adjacent = peers(size)
    candidates = {(r, c): set(range(1, size + 1)) for r in range(size) for c in range(size) if not grid[r][c]}
    trace = []

    def record(rule, action, cell, value):
        r, c = cell
        trace.append({"id": len(trace), "rule": rule, "action": action,
                      "r": r, "c": c, "value": value,
                      "candidates_before": sorted(candidates[cell])})

    def eliminate(cell, value, rule):
        if cell in candidates and value in candidates[cell]:
            record(rule, "eliminate", cell, value)
            candidates[cell].remove(value)
            if not candidates[cell]:
                raise ValueError(f"Contradiction at {cell}")

    def assign(cell, value, rule):
        record(rule, "assign", cell, value)
        r, c = cell
        grid[r][c] = value
        del candidates[cell]
        prefix = "hidden_single" if rule.startswith("hidden_single") else rule
        for other in adjacent[cell]:
            eliminate(other, value, prefix + "_peer_elim")

    for r in range(size):
        for c in range(size):
            if grid[r][c]:
                for other in adjacent[r, c]:
                    eliminate(other, grid[r][c], "initial_elim")

    while candidates:
        before = len(candidates)
        for cell in list(candidates):
            if cell in candidates and len(candidates[cell]) == 1:
                assign(cell, next(iter(candidates[cell])), "naked_single")
        # A full naked-single pass is followed by row/column/block singles.
        for name, family in zip(("row", "col", "block"), units(size)):
            for unit in family:
                for value in range(1, size + 1):
                    locations = [cell for cell in unit if value in candidates.get(cell, ())]
                    if len(locations) == 1:
                        assign(locations[0], value, "hidden_single_" + name)
        if len(candidates) == before:
            raise ValueError("Puzzle needs deductions beyond naked/hidden singles")
    if not valid_solution(grid, size):
        raise ValueError("Solver produced an invalid solution")
    if trace:
        trace[0]["generator"] = "fg-sudoku-release-v1"
    return grid, trace


def trace_cost(trace):
    return sum({"assign": 2, "eliminate": 1}[event["action"]] for event in trace)


def replay_trace(puzzle, trace):
    """Check every recorded operation, then return the completed grid.

    Original traces and recomputed traces share the same operation schema.
    """
    size = len(puzzle)
    grid = [row[:] for row in puzzle]
    candidates = {(r, c): set(range(1, size + 1)) for r in range(size) for c in range(size) if not grid[r][c]}
    adjacent = peers(size)
    all_units = [unit for family in units(size) for unit in family]
    if not isinstance(trace, list):
        raise ValueError("Expected a list of solver operations")
    for index, event in enumerate(trace):
        cell, value = (event["r"], event["c"]), event["value"]
        if event["id"] != index or cell not in candidates or value not in candidates[cell]:
            raise ValueError(f"Invalid trace operation {index}")
        if event["candidates_before"] != sorted(candidates[cell]):
            raise ValueError(f"Incorrect candidate state at operation {index}")
        rule, action = event["rule"], event["action"]
        if action == "eliminate":
            if not any(grid[r][c] == value for r, c in adjacent[cell]):
                raise ValueError(f"Unsupported elimination at operation {index}")
            candidates[cell].remove(value)
            if not candidates[cell]:
                raise ValueError(f"Empty candidate set at operation {index}")
        elif action == "assign":
            if any(grid[r][c] == value for r, c in adjacent[cell]):
                raise ValueError(f"Conflicting assignment at operation {index}")
            if rule == "naked_single":
                justified = len(candidates[cell]) == 1
            elif rule.startswith("hidden_single_"):
                justified = any(
                    cell in unit and [pos for pos in unit if value in candidates.get(pos, ())] == [cell]
                    for unit in all_units
                )
            else:
                justified = False
            if not justified:
                raise ValueError(f"Unjustified assignment at operation {index}")
            grid[cell[0]][cell[1]] = value
            del candidates[cell]
        else:
            raise ValueError(f"Unknown trace action: {action}")
    if candidates or not valid_solution(grid, size):
        raise ValueError("Trace does not finish a valid solution")
    return grid


def count_solutions(puzzle, limit=2):
    """Independent MRV backtracking check; stops after finding two solutions."""
    size = len(puzzle)
    grid = [row[:] for row in puzzle]
    adjacent = peers(size)

    def search():
        choices = []
        for r in range(size):
            for c in range(size):
                if not grid[r][c]:
                    options = set(range(1, size + 1)) - {grid[a][b] for a, b in adjacent[r, c]}
                    if not options:
                        return 0
                    choices.append((len(options), r, c, sorted(options)))
        if not choices:
            return int(valid_solution(grid, size))
        _, r, c, options = min(choices)
        total = 0
        for value in options:
            grid[r][c] = value
            total += search()
            if total >= limit:
                break
        grid[r][c] = 0
        return total

    return search()
