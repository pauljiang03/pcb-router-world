"""Generate a net-ordering distillation dataset (see docs/ROUTING.md section 4.1).

Routing ORDER strongly affects how many nets route conflict-free, so a learned
policy that predicts a good order from board features could replace expensive
multi-start search in one shot. This script builds (per-net features -> best
routing order) examples by running the negotiated router under many orders per
board and recording the best one. CPU only; the model fit itself needs compute.

Run from the repo root:
    python scripts/gen_ordering_data.py --boards 8 --orders 12 --out ordering_data.npz
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

import numpy as np
from envs.board import load_te_example, generate_candidate_grid, check_tp_spacing
from envs.routing import (RoutingGrid, _negotiate, _remove_crossings,
                          TP_CLEARANCE_CELLS)


def planar_placement(b, c, n):
    """Non-crossing angular placement (see docs/ROUTING.md)."""
    ccx = b.connector_x + b.connector_w / 2
    ccy = b.connector_y + b.connector_h / 2
    chosen = []
    for idx in np.argsort(-np.hypot(c[:, 0] - ccx, c[:, 1] - ccy)):
        if len(chosen) >= n:
            break
        if check_tp_spacing(chosen, *c[idx]):
            chosen.append(tuple(c[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(n), key=lambda i: np.arctan2(b.traces[i].start_y - ccy,
                                                     b.traces[i].start_x - ccx))
    placed = [None] * n
    for k, i in enumerate(pins):
        if k < len(tps):
            placed[i] = tps[k]
    return placed


def _setup(board, placed):
    """Replicate route_all_traces' grid setup (blocked + escape carving + pad
    keep-out) so we can evaluate individual routing orders."""
    n = len(placed)
    grid = RoutingGrid(board)
    rows, cols = grid.rows, grid.cols
    blocked = (grid.obstacle_grid > 0).copy()
    cells = []
    for i in range(n):
        t = board.traces[i]
        sc, sr = grid._world_to_grid(t.start_x, t.start_y)
        ec, er = grid._world_to_grid(*placed[i])
        cells.append(((sr, sc), (er, ec)))
        blocked[sr, sc] = False
        blocked[er, ec] = False
    ccx = board.connector_x + board.connector_w / 2
    ccy = board.connector_y + board.connector_h / 2
    crow = grid._world_to_grid(ccx, ccy)[1]
    for (sr, sc), _e in cells:
        d = 1 if sr > crow else -1
        for j in range(6):
            r = sr + d * j
            if 0 <= r < rows:
                blocked[r, sc] = False
    endpoints = set()
    for s, e in cells:
        endpoints.add(s); endpoints.add(e)
    tp_owner = -np.ones((rows, cols), dtype=np.int32)
    rad = TP_CLEARANCE_CELLS
    for i, (_s, (er, ec)) in enumerate(cells):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr, cc = er + dr, ec + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        tp_owner[rr, cc] = i
    for (sr, sc), _e in cells:
        d = 1 if sr > crow else -1
        for j in range(6):
            r = sr + d * j
            if 0 <= r < rows:
                tp_owner[r, sc] = -1
    return grid, rows, cols, blocked, cells, endpoints, tp_owner


def fails_for_order(ctx, order):
    grid, rows, cols, blocked, cells, ep, tp_owner = ctx
    routes = _negotiate(blocked, rows, cols, cells, ep, order, 25, 4.0, 1.0, tp_owner)
    routes = _remove_crossings(routes, cells, rows, cols, grid)
    return sum(1 for r in routes if r is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=8)
    ap.add_argument("--orders", type=int, default=12)
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--out", type=str, default="ordering_data.npz")
    a = ap.parse_args()
    rng = np.random.RandomState(0)

    X, Yrank, default_fails, best_fails = [], [], [], []
    for s in range(a.boards):
        b = load_te_example(num_traces=a.num_traces, seed=s)
        n = len(b.traces)
        c, rc = generate_candidate_grid(b, 6.5)
        c = c[:rc]
        placed = planar_placement(b, c, n)
        ctx = _setup(b, placed)
        base = list(range(n))
        dist = [np.hypot(placed[i][0] - b.traces[i].start_x,
                         placed[i][1] - b.traces[i].start_y) for i in range(n)]
        orders = [base,
                  sorted(base, key=lambda i: -dist[i]),
                  sorted(base, key=lambda i: dist[i])]
        while len(orders) < a.orders:
            orders.append(list(rng.permutation(n)))
        fs = [fails_for_order(ctx, o) for o in orders]
        best = orders[int(np.argmin(fs))]

        ccx = b.connector_x + b.connector_w / 2
        ccy = b.connector_y + b.connector_h / 2
        diag = np.hypot(b.width, b.height)
        feat = np.array([[(b.traces[i].start_x - b.x_min) / b.width,
                          (b.traces[i].start_y - b.y_min) / b.height,
                          (placed[i][0] - b.x_min) / b.width,
                          (placed[i][1] - b.y_min) / b.height,
                          dist[i] / diag,
                          np.arctan2(placed[i][1] - ccy, placed[i][0] - ccx) / np.pi]
                         for i in range(n)], dtype=np.float32)
        rank = np.zeros(n, dtype=np.int32)
        for pos, i in enumerate(best):
            rank[i] = pos
        X.append(feat); Yrank.append(rank)
        default_fails.append(fs[0]); best_fails.append(min(fs))
        print(f"  board {s}: identity-order fails={fs[0]}  best-of-{a.orders} fails={min(fs)}")

    X = np.array(X); Yrank = np.array(Yrank)
    np.savez(a.out, features=X, best_order_rank=Yrank,
             default_fails=np.array(default_fails), best_fails=np.array(best_fails))
    print(f"\nsaved {a.out}: features {X.shape} (boards x nets x 6), ranks {Yrank.shape}")
    print(f"failures — identity order avg {np.mean(default_fails):.2f}  vs  "
          f"best-of-{a.orders} avg {np.mean(best_fails):.2f}  "
          f"(learnable gain {np.mean(np.array(default_fails) - np.array(best_fails)):.2f} nets/board)")
    print("A policy that predicts the best order from features captures this gain in one shot.")


if __name__ == "__main__":
    main()
