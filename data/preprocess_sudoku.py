from typing import Sequence, Tuple, Dict, List, Optional
import numpy as np


def _range_mask(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return (x >= lo) & (x <= hi)


def _infer_layout(
    quads: np.ndarray,
    *,
    expect_strategy_all_zero: bool = False,
    base_options: Tuple[int, ...] = (0, 1),
) -> Tuple[int, int, int, int, int]:
    """
    Infer which positions in a 4-tuple correspond to (row, col, value, strategy).

    Returns: (base, row_pos, col_pos, val_pos, strat_pos)
      base = 0 means rows/cols are 0..8
      base = 1 means rows/cols are 1..9 (we'll later subtract 1)
    """
    n = quads.shape[0]
    if n == 0:
        # No data to infer from; caller should handle separately.
        raise ValueError("Cannot infer layout from empty quad set.")

    best: Optional[Tuple[int, int, int, int, int]] = None
    best_score = -1e18

    for base in base_options:
        rc_lo, rc_hi = (0, 8) if base == 0 else (1, 9)

        for row_pos in range(4):
            for col_pos in range(4):
                if col_pos == row_pos:
                    continue
                for val_pos in range(4):
                    if val_pos in (row_pos, col_pos):
                        continue
                    strat_pos = ({0, 1, 2, 3} - {row_pos, col_pos, val_pos}).pop()

                    rows = quads[:, row_pos]
                    cols = quads[:, col_pos]
                    vals = quads[:, val_pos]
                    strat = quads[:, strat_pos]

                    valid = (
                        _range_mask(rows, rc_lo, rc_hi)
                        & _range_mask(cols, rc_lo, rc_hi)
                        & _range_mask(vals, 1, 9)
                    )
                    valid_count = int(valid.sum())

                    # Strong signal: (row, col) pairs should be unique within a block
                    # (each cell should appear at most once).
                    if valid_count > 0:
                        rr = rows[valid].astype(int)
                        cc = cols[valid].astype(int)
                        if base == 1:
                            rr = rr - 1
                            cc = cc - 1
                        if ((rr < 0) | (rr > 8) | (cc < 0) | (cc > 8)).any():
                            unique_pairs = 0
                        else:
                            unique_pairs = len(set(map(tuple, np.stack([rr, cc], axis=1).tolist())))
                    else:
                        unique_pairs = 0

                    score = valid_count + 3.0 * unique_pairs

                    # For "givens" block, strategy is often all zeros.
                    if expect_strategy_all_zero:
                        score += 20.0 * float(np.all(strat == 0))

                    # Tiny bonus if strategy looks like a "code" (often > 9)
                    if strat.max(initial=0) > 9:
                        score += 0.25

                    if score > best_score:
                        best_score = score
                        best = (base, row_pos, col_pos, val_pos, strat_pos)

    if best is None:
        raise ValueError("Failed to infer quad layout.")
    return best


def _extract_rcv(
    quads: np.ndarray,
    base: int,
    row_pos: int,
    col_pos: int,
    val_pos: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = quads[:, row_pos].astype(int)
    c = quads[:, col_pos].astype(int)
    v = quads[:, val_pos].astype(int)
    if base == 1:
        r = r - 1
        c = c - 1
    return r, c, v


def _find_duplicates(vals: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for x in vals:
        counts[x] = counts.get(x, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def check_givens_do_not_violate_sudoku(puzzle_grid: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Checks Sudoku rule constraints ONLY on the non-zero entries.
    Returns: (ok, violations)
    """
    if puzzle_grid.shape != (9, 9):
        raise ValueError("puzzle_grid must be shape (9,9)")

    violations: List[str] = []

    # Rows
    for r in range(9):
        row = [int(x) for x in puzzle_grid[r, :] if int(x) != 0]
        dups = _find_duplicates(row)
        if dups:
            violations.append(f"Row {r}: duplicate(s) {dups}")

    # Cols
    for c in range(9):
        col = [int(x) for x in puzzle_grid[:, c] if int(x) != 0]
        dups = _find_duplicates(col)
        if dups:
            violations.append(f"Col {c}: duplicate(s) {dups}")

    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    x = int(puzzle_grid[r, c])
                    if x != 0:
                        box.append(x)
            dups = _find_duplicates(box)
            if dups:
                violations.append(f"Box ({br},{bc}): duplicate(s) {dups}")

    return (len(violations) == 0), violations


def sudoku_example_to_162(
    example: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Convert one dataset example vector to:
      - cache_record: puzzle (81) followed by solution (81), row-major
      - meta: puzzle/solution grids + validation info

    The 162 values are a storage record, not a model sequence. At runtime the
    model always operates on one 81-token grid and uses the puzzle half only to
    identify fixed clue positions.

    Assumes the example is:
      [k, ... 324 ints ...] where 324 = 81*4 (81 quads), split into:
        first k quads = givens
        remaining 81-k quads = empties
    """
    a = np.asarray(example, dtype=int).ravel()

    expected_len = 1 + 81 * 4
    if a.size != expected_len:
        raise ValueError(f"Expected length {expected_len} (1 + 81*4), got {a.size}")

    k = int(a[0])
    if not (0 <= k <= 81):
        raise ValueError(f"Invalid givens count k={k}")

    quads = a[1:].reshape(81, 4)
    givens_quads = quads[:k]
    empties_quads = quads[k:]

    # Infer layout from empties if possible (usually more informative)
    if empties_quads.shape[0] > 0:
        base_e, rpe, cpe, vpe, spe = _infer_layout(
            empties_quads, expect_strategy_all_zero=False, base_options=(0, 1)
        )
        base_g, rpg, cpg, vpg, spg = _infer_layout(
            givens_quads if givens_quads.shape[0] > 0 else empties_quads,
            expect_strategy_all_zero=True,
            base_options=(base_e,),
        )
    else:
        # All cells are givens; infer from givens only
        base_g, rpg, cpg, vpg, spg = _infer_layout(
            givens_quads, expect_strategy_all_zero=True, base_options=(0, 1)
        )
        base_e, rpe, cpe, vpe, spe = base_g, rpg, cpg, vpg, spg

    puzzle = np.zeros((9, 9), dtype=int)
    solution = np.zeros((9, 9), dtype=int)

    # Fill givens into puzzle + solution
    if k > 0:
        gr, gc, gv = _extract_rcv(givens_quads, base_g, rpg, cpg, vpg)
        for rr, cc, vv in zip(gr.tolist(), gc.tolist(), gv.tolist()):
            if not (0 <= rr <= 8 and 0 <= cc <= 8):
                raise ValueError(f"Given cell coordinate out of range: (r,c)=({rr},{cc})")
            if not (1 <= vv <= 9):
                raise ValueError(f"Given value out of range at (r,c)=({rr},{cc}): {vv}")
            if puzzle[rr, cc] not in (0, vv):
                raise ValueError(f"Conflicting given at (r,c)=({rr},{cc}): {puzzle[rr,cc]} vs {vv}")
            puzzle[rr, cc] = vv
            solution[rr, cc] = vv

    # Fill empties into solution
    if empties_quads.shape[0] > 0:
        er, ec, ev = _extract_rcv(empties_quads, base_e, rpe, cpe, vpe)
        for rr, cc, vv in zip(er.tolist(), ec.tolist(), ev.tolist()):
            if not (0 <= rr <= 8 and 0 <= cc <= 8):
                raise ValueError(f"Empty-cell coordinate out of range: (r,c)=({rr},{cc})")
            if not (1 <= vv <= 9):
                raise ValueError(f"Solution value out of range at (r,c)=({rr},{cc}): {vv}")
            if solution[rr, cc] not in (0, vv):
                raise ValueError(
                    f"Conflicting solution at (r,c)=({rr},{cc}): {solution[rr,cc]} vs {vv}"
                )
            solution[rr, cc] = vv

    # Validate givens don't violate Sudoku constraints
    givens_ok, givens_violations = check_givens_do_not_violate_sudoku(puzzle)

    # Create the cache record (row-major)
    prompt81 = puzzle.reshape(-1)          # 0 for empty
    answer81 = solution.reshape(-1)        # should be 1..9 everywhere (if dataset is complete)
    cache_record = np.concatenate([prompt81, answer81], axis=0)

    meta: Dict[str, object] = {
        "givens_count": k,
        "puzzle_grid": puzzle,
        "solution_grid": solution,
        "givens_ok": givens_ok,
        "givens_violations": givens_violations,
        "layout_givens": {"base": base_g, "row_pos": rpg, "col_pos": cpg, "val_pos": vpg, "strat_pos": spg},
        "layout_empties": {"base": base_e, "row_pos": rpe, "col_pos": cpe, "val_pos": vpe, "strat_pos": spe},
    }
    return cache_record, meta
