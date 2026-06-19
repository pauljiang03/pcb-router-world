"""Challenging board (obstacle 'moat' — traces can't fan straight out), routed with
GUARANTEED trace-to-trace clearance.

- `challenge_board`: a moat of keep-outs rings the connector with only a few gaps, so
  traces must funnel through them (forces detours + usually a 2nd layer).
- Rectilinear base (`route_auto_layers(diagonal=False)`): axis-adjacent cells are
  exactly the trace pitch, so there is no 45-degree parallel-diagonal gap (which would
  be only pitch/sqrt(2) ~= 0.94mm < pitch). Layers are assigned by routability.
- `any_angle_shortcut` per layer: straightens to ANY-ANGLE only where the segment stays
  >= the pitch from same-layer traces (exact distance) -> shorter, clearance preserved.

Every same-layer pair stays >= the pitch (`min_trace_separation`), with 0 same-layer
crossings. Run from the repo root:  python scripts/route_any_angle.py [--gaps K]
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import challenge_board
from envs.routing import (route_auto_layers, any_angle_shortcut, count_crossings,
                          min_trace_separation, CELL_SIZE, TP_CLEARANCE_CELLS)


def _len(P):
    return sum(sum(np.hypot(p[k+1][0]-p[k][0], p[k+1][1]-p[k][1]) for k in range(len(p)-1))
               for p in P if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--board_size", type=float, default=120.0)
    ap.add_argument("--gaps", type=int, default=3)
    ap.add_argument("--out", type=str, default="eval_results/router_any_angle.png")
    a = ap.parse_args()
    n = a.num_traces

    board, placed = challenge_board(a.board_size, n, a.gaps)
    paths, L, lof, f, lx = route_auto_layers(board, placed, max_layers=6, diagonal=False)
    used = sorted(set(l for l in lof if l >= 0))

    aa = list(paths)                                          # any-angle, per layer
    for layer in used:
        idx = [i for i in range(n) if lof[i] == layer]
        sc = any_angle_shortcut([paths[i] for i in idx], board)
        for k, i in enumerate(idx):
            aa[i] = sc[k]

    def per(fn, P):
        return min((fn([P[i] for i in range(n) if lof[i] == layer]) for layer in used),
                   default=float('inf'))
    def per_cross(P):
        return max((count_crossings([P[i] for i in range(n) if lof[i] == layer]) for layer in used),
                   default=0)

    print(f"challenge moat board {a.board_size:.0f}mm, {a.gaps} gaps, {n} traces (pitch {CELL_SIZE:.2f}mm)")
    print(f"  routed {n-f}/{n}  layers {len(used)}  vias {sum(1 for l in lof if l >= 1)}  "
          f"same-layer crossings {per_cross(aa)}")
    print(f"  per-layer min trace separation: base {per(min_trace_separation, paths):.2f} -> "
          f"any-angle {per(min_trace_separation, aa):.2f}mm  (must be >= {CELL_SIZE:.2f})")
    print(f"  total length: any-angle {_len(aa):.0f}mm")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, aa, a.out, labels=True, legend=True,
            path_layers=lof, keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE,
            title=(f"Challenge moat ({a.gaps} gaps) + clearance-ensured any-angle: {n-f}/{n} routed, "
                   f"{len(used)} layers, min trace sep {per(min_trace_separation, aa):.2f}mm >= pitch {CELL_SIZE:.2f}"))
        print("figure:", a.out)
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
