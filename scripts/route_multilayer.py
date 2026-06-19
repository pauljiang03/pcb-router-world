"""Multi-layer routing with AUTOMATIC layer assignment and spread endpoints.

`route_auto_layers` routes what fits planar on layer 0, pushes the nets that can't
route conflict-free onto the next layer, and repeats. The non-routing zone blocks
EVERY layer; same-layer crossings are forbidden; inter-layer crossings are vias.
The layer count is found by routability, so a good (spread) placement routes on a
single layer and only denser placements need more layers. Endpoints are spread over
the board (`spread_placement`), not bucketed into per-layer rows. Routed traces are
then length-matched.

Run from the repo root:  python scripts/route_multilayer.py [--seed N]
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import load_te_example, spread_placement
from envs.routing import (route_auto_layers, equalize_lengths, count_crossings,
                          validate_routing_constraints, TP_CLEARANCE_CELLS, CELL_SIZE)


def _spread(vals):
    fin = [x for x in vals if x < float("inf")]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--out", type=str, default="eval_results/router_multilayer.png")
    a = ap.parse_args()
    n = a.num_traces

    board = load_te_example(num_traces=n, seed=a.seed)
    placed = spread_placement(board, n)              # endpoints spread over the board
    paths, L, layer_of, fails, lx = route_auto_layers(board, placed)
    used = sorted(set(l for l in layer_of if l >= 0))
    vias = sum(1 for l in layer_of if l >= 1)

    # Length-match across all layers to one global target (same NRZ-blocking board).
    breakout = board.traces[0].breakout_length
    target_mm = max(x for x in L if x < float("inf")) - breakout
    eq = [None] * n
    eqL = [float("inf")] * n
    for layer in used:
        idxs = [i for i in range(n) if layer_of[i] == layer]
        ep, el, _, _ = equalize_lengths(board, [paths[i] for i in idxs], target_mm=target_mm)
        for k, i in enumerate(idxs):
            eq[i] = ep[k]
            eqL[i] = el[k]
    v = validate_routing_constraints(board, eq)

    print(f"routed {n - fails}/{n}  layers {used}  vias {vias}  same-layer crossings {lx}  "
          f"pad-clearance {v['tp_to_trace_min']:.1f}mm")
    print(f"  length spread {_spread(L):.2f} -> equalized {_spread(eqL):.2f}  "
          f"total crossings {count_crossings(eq)}")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, eq, a.out, labels=True, legend=True,
            path_layers=layer_of, keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE,
            title=(f"Auto multi-layer + spread endpoints: {n - fails}/{n} routed, "
                   f"{len(used)} layer(s), {vias} vias, {count_crossings(eq)} crossings "
                   f"(inner layers dashed)"))
        print("figure:", a.out)
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
