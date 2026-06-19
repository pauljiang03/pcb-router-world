"""Multi-via / layer budget for the parametric moat board.

How many copper layers (and vias) does it take to route ALL traces, as a function of
TRACES-PER-GAP? The single-layer bottleneck is topological — traces funnelling through
K gaps — so this sweep lets you pick a board spec (traces, gaps, size) that hits a
target layer count. Reads the parametric ChallengeSpec, so every knob is adjustable.

Run from the repo root:
  python scripts/layer_budget.py --board_size 120 --gaps 2 3 4 --traces 12 16 20 24
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import make_challenge, ChallengeSpec
from envs.routing import route_auto_layers, count_crossings, min_trace_separation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board_size", type=float, default=120.0)
    ap.add_argument("--gaps", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--traces", type=int, nargs="+", default=[12, 16, 20, 24])
    ap.add_argument("--placement", choices=["ring", "gap_aligned"], default="ring")
    ap.add_argument("--diagonal", type=int, default=1, help="1=45deg base, 0=rectilinear (clearance-safe)")
    ap.add_argument("--max_layers", type=int, default=8)
    a = ap.parse_args()

    print(f"board {a.board_size:.0f}mm, placement={a.placement}, "
          f"{'45deg' if a.diagonal else 'rectilinear'} base")
    print(f"{'traces':>6} {'gaps':>5} {'tr/gap':>7} {'routed':>7} {'layers':>7} {'vias':>5} "
          f"{'same-layer x':>12} {'min sep':>8}")
    for n in a.traces:
        for g in a.gaps:
            b, pl = make_challenge(ChallengeSpec(
                board_size=a.board_size, num_traces=n, n_gaps=g, placement=a.placement))
            p, L, lof, f, lx = route_auto_layers(b, pl, max_layers=a.max_layers,
                                                 diagonal=bool(a.diagonal))
            used = sorted(set(l for l in lof if l >= 0))
            vias = sum(1 for l in lof if l >= 1)
            xs = max((count_crossings([p[i] for i in range(n) if lof[i] == Lr]) for Lr in used),
                     default=0)
            sep = min((min_trace_separation([p[i] for i in range(n) if lof[i] == Lr]) for Lr in used),
                      default=0.0)
            print(f"{n:>6} {g:>5} {n/g:>7.1f} {f'{n-f}/{n}':>7} {len(used):>7} {vias:>5} "
                  f"{xs:>12} {sep:>8.2f}")


if __name__ == "__main__":
    main()
