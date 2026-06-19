"""Route every trace to the TOP of the board, then length-equalize (exploratory).

--rows 1: single pin row low on the board, all escaping upward (routes 20/20).
--rows 2: two-row connector — the LOWER row escapes downward and wraps up the
          sides, the UPPER row escapes straight up. The single-layer wrap is hard,
          so not all 20 route, but every routed trace is then length-matched
          (envs.routing.equalize_lengths) so the figure shows serpentine meanders
          bringing them all to equal length.

Run from the repo root:  python scripts/route_to_top.py --rows 2
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import (load_edge_board, fan_to_top_placement,
                        load_edge_board_2row, wrap_to_top_placement)
from envs.routing import (route_all_traces, count_crossings, equalize_lengths,
                          validate_routing_constraints, TP_CLEARANCE_CELLS, CELL_SIZE)


def _spread(lengths):
    fin = [x for x in lengths if x < float("inf")]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--rows", type=int, default=2, choices=[1, 2])
    ap.add_argument("--out", type=str, default="eval_results/router_top_fan.png")
    a = ap.parse_args()

    if a.rows == 2:
        board = load_edge_board_2row(a.num_traces)
        placed = wrap_to_top_placement(board, a.num_traces)
    else:
        board = load_edge_board(a.num_traces, 180.0, 180.0)
        placed = fan_to_top_placement(board, a.num_traces)

    paths, lengths, fails = route_all_traces(board, placed)
    routed = a.num_traces - fails
    s0 = _spread(lengths)

    # Stage 3: length-match every routed trace to the longest.
    eq_paths, eq_lengths, target, matched = equalize_lengths(board, paths)
    eqfin = [x for x in eq_lengths if x < float("inf")]
    s1 = _spread(eq_lengths)
    v = validate_routing_constraints(board, eq_paths)

    print(f"{a.rows}-row edge board -> top: routed {routed}/{a.num_traces}, "
          f"crossings {count_crossings(eq_paths)}, pad-clearance {v['tp_to_trace_min']:.1f}mm")
    print(f"  length spread {s0:.2f} -> equalized {s1:.2f} "
          f"(target ~{target:.0f}mm, matched {matched}/{routed})")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, eq_paths, a.out, labels=True, legend=True,
            keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE,
            title=(f"{a.rows}-row edge -> top, length-matched: {routed}/{a.num_traces} routed, "
                   f"{count_crossings(eq_paths)} crossings, equal length ~{target:.0f}mm (spread {s1:.2f})"))
        print("figure:", a.out)
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
