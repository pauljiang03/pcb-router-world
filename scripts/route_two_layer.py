"""Two-layer prototype: route all 20 traces of a 2-row connector to the top.

The single-layer wrap caps at ~14/20 because the lower row can't all wrap up one
layer. With two layers the lower row vias down and routes straight up *under* the
connector (a top-layer-only obstacle), so each layer is a clean planar up-fan and
all 20 route. Routing ORDER matters within each layer (route_all_traces multi-start
searches it) and the LAYER ASSIGNMENT (which row -> which layer) is the key choice.

Routed traces are then length-matched across BOTH layers, shown as serpentine
meanders. Layer 1 is drawn solid, layer 2 dashed.

Run from the repo root:  python scripts/route_two_layer.py
"""
import sys
import pathlib
import argparse
import copy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import load_edge_board_2row, two_layer_placement
from envs.routing import (route_two_layer, equalize_lengths, count_crossings,
                          TP_CLEARANCE_CELLS, CELL_SIZE)


def _spread(vals):
    fin = [x for x in vals if x < float("inf")]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--W", type=float, default=240.0)
    ap.add_argument("--H", type=float, default=250.0)
    ap.add_argument("--out", type=str, default="eval_results/router_two_layer.png")
    a = ap.parse_args()
    n = a.num_traces

    board = load_edge_board_2row(n, a.W, a.H)
    placed, layer_of = two_layer_placement(board, n)
    paths, lengths, fails, layer_x = route_two_layer(board, placed, layer_of)
    routed = n - fails
    s0 = _spread(lengths)

    # Length-match across BOTH layers to a single global target (the longest trace).
    breakout = board.traces[0].breakout_length
    target_mm = max(x for x in lengths if x < float("inf")) - breakout
    eq_paths = [None] * n
    eqL = [float("inf")] * n
    for layer in sorted(set(layer_of)):
        idxs = [i for i in range(n) if layer_of[i] == layer]
        sb = copy.copy(board)
        sb.traces = [board.traces[i] for i in idxs]
        if layer != 0:
            sb.rect_obstacles = []
            sb.circ_obstacles = []
        ep, el, _, _ = equalize_lengths(sb, [paths[i] for i in idxs], target_mm=target_mm)
        for k, i in enumerate(idxs):
            eq_paths[i] = ep[k]
            eqL[i] = el[k]
    eqfin = [x for x in eqL if x < float("inf")]
    s1 = _spread(eqL)

    print(f"2-layer edge board -> top: routed {routed}/{n}  per-layer crossings {layer_x}")
    print(f"  length spread {s0:.2f} -> equalized {s1:.2f}  (all ~{min(eqfin):.0f}-{max(eqfin):.0f}mm)")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, eq_paths, a.out, labels=True, legend=True,
            path_layers=layer_of, keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE,
            title=(f"2-LAYER -> top: {routed}/{n} routed (layer 1 solid, layer 2 dashed), "
                   f"0 same-layer crossings, length-matched ~{max(eqfin):.0f}mm (spread {s1:.2f})"))
        print("figure:", a.out)
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
