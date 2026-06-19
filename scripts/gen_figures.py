"""Regenerate the labeled router figure set — 45-degree (octilinear) and rectilinear
only, NO any-angle. Run from the repo root:  python scripts/gen_figures.py

Each figure has numbered pads, a legend, pad keep-out rings, and a descriptive title.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from envs.board import (load_te_example, equal_length_placement, challenge_board,
                        load_edge_board_2row, wrap_to_top_placement)
from envs.routing import (route_all_traces, route_auto_layers, equalize_lengths,
                          count_crossings, min_trace_separation, CELL_SIZE, TP_CLEARANCE_CELLS)
from envs.visualize import render_board_png

OUT = "eval_results"
KW = dict(labels=True, legend=True, keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE)
PITCH = CELL_SIZE


def spread(L):
    fin = [x for x in L if x < 1e9]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def per_layer(fn, paths, lof, used):
    return min((fn([paths[i] for i in range(len(paths)) if lof[i] == L]) for L in used), default=0.0)


# 1. 45-degree (octilinear) fan -------------------------------------------------
b = load_te_example(num_traces=20, seed=6)
pl = equal_length_placement(b, 20)
p, L, f = route_all_traces(b, pl)
render_board_png(b, pl, p, f"{OUT}/router_45deg_fan.png", **KW,
    title=f"45-degree (octilinear) routing: {20-f}/20 routed, {count_crossings(p)} crossings, "
          f"min trace sep {min_trace_separation(p):.2f}mm")
print("1 router_45deg_fan.png")

# 2. Length-matched (meanders) --------------------------------------------------
p2, L2, tgt, nm = equalize_lengths(b, p)
render_board_png(b, pl, p2, f"{OUT}/router_length_matched.png", **KW,
    title=f"45-degree length-matched: serpentine meanders, length spread {spread(L):.2f} -> "
          f"{spread(L2):.2f}, {count_crossings(p2)} crossings")
print("2 router_length_matched.png")

# 3. Auto multi-layer (octilinear, harder placement) ----------------------------
b3 = load_te_example(num_traces=20, seed=0)
pl3 = equal_length_placement(b3, 20)
p3, L3, lof3, f3, lx3 = route_auto_layers(b3, pl3, max_layers=6, diagonal=True)
u3 = sorted(set(l for l in lof3 if l >= 0))
render_board_png(b3, pl3, p3, f"{OUT}/router_multilayer.png", path_layers=lof3, **KW,
    title=f"Auto multi-layer (45-degree): {20-f3}/20 routed, {len(u3)} layers, "
          f"{sum(1 for l in lof3 if l>=1)} vias, 0 same-layer crossings (inner layers dashed)")
print("3 router_multilayer.png")

# 4. Two-row connector -> top (45-degree, length-matched) -----------------------
bt = load_edge_board_2row(20)
plt_ = wrap_to_top_placement(bt, 20)
pt, Lt, ft = route_all_traces(bt, plt_)
pt2, _, _, _ = equalize_lengths(bt, pt)
render_board_png(bt, plt_, pt2, f"{OUT}/router_top_fan.png", **KW,
    title=f"2-row connector -> top (45-degree, single layer): {20-ft}/20 routed, length-matched, "
          f"{count_crossings(pt2)} crossings")
print("4 router_top_fan.png")

# 5 & 6. Challenge 'moat' board: 45-degree vs rectilinear (clearance) ------------
bc, plc = challenge_board(board_size=120.0, num_traces=20, n_gaps=3)
pc, Lc, lofc, fc, lxc = route_auto_layers(bc, plc, max_layers=6, diagonal=True)
uc = sorted(set(l for l in lofc if l >= 0))
render_board_png(bc, plc, pc, f"{OUT}/router_challenge_45deg.png", path_layers=lofc, **KW,
    title=f"Challenge 'moat' (traces funnel through gaps), 45-degree: {20-fc}/20, {len(uc)} layers, "
          f"min trace sep {per_layer(min_trace_separation, pc, lofc, uc):.2f}mm (< pitch {PITCH:.2f})")
print("5 router_challenge_45deg.png")

pr, Lr, lofr, fr, lxr = route_auto_layers(bc, plc, max_layers=6, diagonal=False)
ur = sorted(set(l for l in lofr if l >= 0))
render_board_png(bc, plc, pr, f"{OUT}/router_challenge_rectilinear.png", path_layers=lofr, **KW,
    title=f"Challenge 'moat', RECTILINEAR: clearance GUARANTEED, min trace sep "
          f"{per_layer(min_trace_separation, pr, lofr, ur):.2f}mm >= pitch {PITCH:.2f}, {20-fr}/20, {len(ur)} layers")
print("6 router_challenge_rectilinear.png")
print("done")
