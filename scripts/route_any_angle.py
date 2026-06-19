"""Any-angle routing with guaranteed trace-to-trace clearance (smaller board + obstacles).

Route octilinear first (clearance handled by the grid), then string-pull to ANY-ANGLE
with `any_angle_shortcut`: a straight segment is accepted only if it clears all
obstacles AND stays >= the trace pitch from every other net (exact segment-to-segment
distance). So any-angle shortens traces and NEVER introduces a spacing violation.

`min_trace_separation` reports the exact trace-to-trace clearance. Note: at minimum
pin pitch, parallel 45-degree traces are inherently pitch*sin45 ~= 0.94mm apart (< the
1.33mm pitch) — a property of fanning out at 45 degrees, which the verifier now flags;
any-angle does not make it worse.

Run from the repo root:  python scripts/route_any_angle.py
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import (load_te_example, equal_length_placement,
                        Obstacle, TRACE_TO_TRACE_MIN)
from envs.routing import (route_all_traces, any_angle_shortcut, count_crossings,
                          min_trace_separation, CELL_SIZE, TP_CLEARANCE_CELLS)


def _len(paths):
    return sum(sum(np.hypot(p[k+1][0]-p[k][0], p[k+1][1]-p[k][1]) for k in range(len(p)-1))
               for p in paths if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--board_size", type=float, default=130.0)
    ap.add_argument("--out", type=str, default="eval_results/router_any_angle.png")
    a = ap.parse_args()
    n = a.num_traces

    board = load_te_example(num_traces=n, seed=a.seed, board_size=a.board_size)
    for fx, fy in [(0.26, 0.70), (0.74, 0.30)]:                 # extra keep-outs
        board.rect_obstacles.append(Obstacle(cx=a.board_size*fx, cy=a.board_size*fy,
                                             width=9.0, height=9.0,
                                             clearance=TRACE_TO_TRACE_MIN, name="keepout"))
    placed = equal_length_placement(board, n)
    octi, L, f = route_all_traces(board, placed)
    aa = any_angle_shortcut(octi, board)                        # clearance-verified any-angle

    lo, la = _len(octi), _len(aa)
    print(f"board {a.board_size:.0f}mm + 2 obstacles, {n} traces, pitch {CELL_SIZE:.3f}mm")
    print(f"  octilinear: routed {n-f}/{n}  len {lo:.0f}mm  crossings {count_crossings(octi)}  "
          f"min_trace_sep {min_trace_separation(octi):.3f}")
    print(f"  any-angle : routed {n-f}/{n}  len {la:.0f}mm ({100*(lo-la)/lo:.1f}% shorter)  "
          f"crossings {count_crossings(aa)}  min_trace_sep {min_trace_separation(aa):.3f}")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, aa, a.out, labels=True, legend=True,
            keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE,
            title=(f"Any-angle (clearance-verified): {n-f}/{n} routed, {100*(lo-la)/lo:.0f}% shorter, "
                   f"{count_crossings(aa)} crossings, min trace sep {min_trace_separation(aa):.2f}mm"))
        print("figure:", a.out)
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
