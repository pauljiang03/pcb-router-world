"""Route every trace to the TOP of the board (exploratory).

Uses load_edge_board (a single pin row low on the board, escaping upward) and a
planar fan_to_top_placement, then routes with the planar router and renders a
labeled figure. Not a general constraint — a demonstration of what the router can
do for a one-sided (edge-connector -> opposite-edge) layout.

Run from the repo root:  python scripts/route_to_top.py
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from envs.board import load_edge_board, fan_to_top_placement
from envs.routing import route_all_traces, count_crossings, validate_routing_constraints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--W", type=float, default=180.0)
    ap.add_argument("--H", type=float, default=180.0)
    ap.add_argument("--out", type=str, default="eval_results/router_top_fan.png")
    a = ap.parse_args()

    board = load_edge_board(a.num_traces, a.W, a.H)
    placed = fan_to_top_placement(board, a.num_traces)
    paths, lengths, fails = route_all_traces(board, placed)
    fin = [x for x in lengths if x < float("inf")]
    v = validate_routing_constraints(board, paths)
    print(f"edge board {a.W:.0f}x{a.H:.0f}, {a.num_traces} traces -> "
          f"routed {a.num_traces - fails}/{a.num_traces}, crossings {count_crossings(paths)}, "
          f"length {sum(fin):.0f}mm, pad-clearance {v['tp_to_trace_min']:.1f}mm")

    try:
        from envs.visualize import render_board_png
        render_board_png(
            board, placed, paths, a.out, labels=True, legend=True,
            title=(f"All traces routed to the TOP: {a.num_traces - fails}/{a.num_traces} "
                   f"routed, {count_crossings(paths)} crossings, {sum(fin):.0f}mm"))
        print("figure:", a.out)
    except Exception as e:  # PIL should always be available; be defensive
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
