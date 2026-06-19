"""
Unit tests for the PCB environment, board, and A* router.

Pure numpy + gymnasium — no matplotlib, no torch, no training. Fast.
Run with:  python -m pytest test_pcb.py -q
       or:  python test_pcb.py
"""
import numpy as np

from envs.board import (
    load_te_example, generate_candidate_grid, check_tp_spacing,
    _is_valid_tp_position, MAX_CANDIDATES, TP_TO_TP_MIN, TP_TO_EDGE_MIN,
)
from envs.routing import route_all_traces, validate_routing_constraints, count_crossings
from envs.pcb_env import TPPlacementEnv


# ---------------------------------------------------------------- board / grid

def test_candidate_grid_fixed_size_and_real_count():
    board = load_te_example(num_traces=10, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    assert cand.shape == (MAX_CANDIDATES, 2)
    assert 0 < real <= MAX_CANDIDATES


def test_all_real_candidates_are_valid_positions():
    board = load_te_example(num_traces=10, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    assert all(_is_valid_tp_position(board, x, y) for x, y in cand[:real])


def test_check_tp_spacing():
    assert check_tp_spacing([(0.0, 0.0)], 0.0, TP_TO_TP_MIN + 1.0)      # far enough
    assert not check_tp_spacing([(0.0, 0.0)], 0.0, TP_TO_TP_MIN - 1.0)  # too close


def test_edge_clearance_enforced():
    board = load_te_example(num_traces=4, seed=0)
    # A point one mm inside the edge violates the TP-to-edge minimum.
    assert not _is_valid_tp_position(board, board.x_min + 1.0, board.y_min + 30.0)
    # A point well inside (away from obstacles) should be valid.
    assert _is_valid_tp_position(board, board.x_min + TP_TO_EDGE_MIN + 20.0,
                                 board.y_min + TP_TO_EDGE_MIN + 20.0)


# ---------------------------------------------------------------------- router

def test_astar_routes_a_greedy_placement():
    board = load_te_example(num_traces=8, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    placed = []
    for t in board.traces:
        d = np.hypot(cand[:, 0] - t.start_x, cand[:, 1] - t.start_y)
        for idx in np.argsort(d):
            if check_tp_spacing(placed, *cand[idx]):
                placed.append(tuple(cand[idx])); break
    paths, lengths, failures = route_all_traces(board, placed)
    assert len(paths) == len(board.traces)
    assert failures < len(board.traces)                 # not everything fails
    assert any(l < float('inf') for l in lengths)       # some real lengths


def test_validate_returns_expected_schema():
    board = load_te_example(num_traces=6, seed=1)
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    placed = [tuple(cand[i]) for i in range(0, 6)]
    paths, _, _ = route_all_traces(board, placed)
    v = validate_routing_constraints(board, paths)
    for key in ("violations", "trace_to_trace_min", "trace_to_edge_min",
                "trace_to_obstacle_min", "all_valid"):
        assert key in v
    assert isinstance(v["all_valid"], bool)


# ------------------------------------------------------------------------- env

def test_reset_regenerates_board_per_seed():
    """P0.1: different seeds -> different layouts; same seed -> reproducible."""
    env = TPPlacementEnv(num_traces=6, seed=1)
    env.reset(seed=1); a = env.board.rect_obstacles[0].cx
    env.reset(seed=2); b = env.board.rect_obstacles[0].cx
    env.reset(seed=1); a2 = env.board.rect_obstacles[0].cx
    assert a != b          # randomization actually takes effect
    assert a == a2         # deterministic for a given seed


def test_invalid_action_never_places_corner_tp():
    """P0.2: an invalid (spacing-masked) action is snapped to a valid candidate,
    never the junk board corner."""
    env = TPPlacementEnv(num_traces=4, seed=3)
    env.reset(seed=3)
    env.step(0)                                   # place one TP
    masked = [i for i in range(env._real_count) if not env.candidate_mask[i]]
    assert masked, "expected some candidates masked out after a placement"
    env.step(masked[0])                           # pick an invalid candidate
    placed = env.placed_tps[-1]
    assert placed != (env.board.x_min, env.board.y_min)   # not the junk corner
    reals = {tuple(env.candidates[i]) for i in range(env._real_count)}
    assert tuple(placed) in reals                 # snapped to a real candidate


def test_env_uses_astar_by_default():
    """P0.4: A* is the default router."""
    assert TPPlacementEnv(num_traces=4, seed=0).use_freerouting is False
    from envs.dreamer_wrapper import PCBDreamerEnv
    assert PCBDreamerEnv(num_traces=4, seed=0)._inner.use_freerouting is False


def test_episode_exposes_reward_components():
    """P1.1: per-component reward breakdown is logged in info."""
    env = TPPlacementEnv(num_traces=4, seed=5)
    env.reset(seed=5)
    info = {}
    for _ in range(env.num_traces):
        valid = np.where(env.candidate_mask[:env._real_count])[0]
        a = int(valid[0]) if len(valid) else 0
        _, _, term, _, info = env.step(a)
    assert term
    assert "reward_components" in info
    assert "routable" in info["reward_components"]


def test_training_env_wrapper_chain():
    """The dreamerv3 wrapper stack (OneHotAction/TimeLimit/SelectAction/UUID)
    used by train.py steps through a full episode and returns reward components."""
    from envs.dreamer_wrapper import PCBDreamerEnv
    from envs import wrappers
    env = PCBDreamerEnv(num_traces=5, seed=7)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, 5)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    obs = env.reset()
    assert "image" in obs
    n = env.action_space.shape[0]      # OneHotAction Box: shape (num_candidates,)
    done, info, steps = False, {}, 0
    while not done and steps < 10:
        onehot = np.zeros(n, dtype=np.float32)
        onehot[np.random.randint(n)] = 1.0
        obs, reward, done, info = env.step({"action": onehot})
        steps += 1
    assert done
    assert "reward_components" in info


def test_pad_keepout_from_other_traces():
    """Each test pad keeps a keep-out clear of every OTHER trace's body (endpoint-
    to-other-trace-body clearance), well beyond the 1-cell trace-to-trace minimum."""
    from envs.routing import validate_routing_constraints, CELL_SIZE
    b = load_te_example(num_traces=16, seed=1)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    ccx = b.connector_x + b.connector_w / 2; ccy = b.connector_y + b.connector_h / 2
    chosen = []
    for idx in np.argsort(-np.hypot(c[:, 0] - ccx, c[:, 1] - ccy)):
        if len(chosen) >= 16: break
        if check_tp_spacing(chosen, *c[idx]): chosen.append(tuple(c[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(16), key=lambda i: np.arctan2(b.traces[i].start_y - ccy,
                                                      b.traces[i].start_x - ccx))
    placed = [None] * 16
    for k, i in enumerate(pins):
        placed[i] = tps[k]
    paths, _, _ = route_all_traces(b, placed)
    v = validate_routing_constraints(b, paths)
    assert v["tp_to_trace_min"] >= 1.5 * CELL_SIZE   # pad keep-out (> 1 cell)


def test_length_matching_is_crossing_safe_and_equalizes():
    """Stage 3 (equalize_lengths) meanders shorter traces toward the longest
    without ever introducing a crossing, and does not worsen length spread."""
    from envs.routing import equalize_lengths
    b = load_te_example(num_traces=20, seed=1)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    ccx = b.connector_x + b.connector_w / 2; ccy = b.connector_y + b.connector_h / 2
    chosen = []
    for idx in np.argsort(-np.hypot(c[:, 0] - ccx, c[:, 1] - ccy)):
        if len(chosen) >= 20: break
        if check_tp_spacing(chosen, *c[idx]): chosen.append(tuple(c[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(20), key=lambda i: np.arctan2(b.traces[i].start_y - ccy,
                                                      b.traces[i].start_x - ccx))
    placed = [None] * 20
    for k, i in enumerate(pins):
        placed[i] = tps[k]
    paths, L, _ = route_all_traces(b, placed)
    fin0 = [x for x in L if x < float('inf')]
    s0 = (max(fin0) - min(fin0)) / np.mean(fin0)
    paths2, L2, _, _ = equalize_lengths(b, paths)
    fin1 = [x for x in L2 if x < float('inf')]
    s1 = (max(fin1) - min(fin1)) / np.mean(fin1)
    assert count_crossings(paths2) == 0          # meander never crosses
    assert s1 <= s0 + 0.05                        # equalizes (never worsens spread)


def test_edge_board_routes_to_top():
    """Connector low + all traces routed UP to test points in the upper region
    (planar fan): every TP is above every start, nearly all route, 0 crossings."""
    from envs.board import load_edge_board, fan_to_top_placement
    b = load_edge_board(num_traces=20, board_w=180.0, board_h=180.0)
    placed = fan_to_top_placement(b, 20)
    start_top = max(t.start_y for t in b.traces)
    assert all(p[1] > start_top for p in placed)      # all traces end above the start
    paths, _, f = route_all_traces(b, placed)
    assert count_crossings(paths) == 0
    assert (20 - f) >= 18                              # nearly all (measured 20/20)


def test_two_row_edge_wrap_and_equalize():
    """2-row edge board: lower row escapes down and wraps up, upper row escapes up.
    The single-layer wrap caps below all, but routed traces are crossing-free and
    the Stage-3 meander reduces their length spread (equalization)."""
    from envs.board import load_edge_board_2row, wrap_to_top_placement
    from envs.routing import equalize_lengths
    b = load_edge_board_2row(num_traces=12, board_w=220.0, board_h=230.0)
    placed = wrap_to_top_placement(b, 12)
    start_top = max(t.start_y for t in b.traces)
    assert all(p[1] > start_top for p in placed)      # all TPs above the connector
    paths, L, f = route_all_traces(b, placed)
    assert count_crossings(paths) == 0
    assert (12 - f) >= 7                                # most route (single-layer wrap is hard)
    fin0 = [x for x in L if x < float('inf')]
    s0 = (max(fin0) - min(fin0)) / np.mean(fin0)
    eq_paths, eqL, _, _ = equalize_lengths(b, paths)
    fin1 = [x for x in eqL if x < float('inf')]
    s1 = (max(fin1) - min(fin1)) / np.mean(fin1)
    assert count_crossings(eq_paths) == 0              # equalization stays planar
    assert s1 < s0                                      # length-matching reduces spread


def test_auto_layers_planar_per_layer_and_respects_nrz():
    """route_auto_layers invariants: every layer is planar (0 same-layer crossings),
    no routed point passes through the non-routing-zone body (obstacles block every
    layer), and a spread placement routes (nearly) all nets."""
    from envs.board import load_te_example, spread_placement
    from envs.routing import route_auto_layers
    b = load_te_example(num_traces=12, seed=6)
    placed = spread_placement(b, 12)
    paths, L, layer_of, f, lx = route_auto_layers(b, placed, max_layers=4)
    assert all(x == 0 for x in lx.values())              # every layer planar (key invariant)
    assert (12 - f) >= 10                                 # spread placement routes (nearly) all

    nrz = [o for o in b.rect_obstacles if o.name == "non_routing_zone"]
    def deep_in_nrz(x, y):                                # inside the body (not boundary rounding)
        return any(abs(x - o.cx) <= o.width / 2 - 1.5 and abs(y - o.cy) <= o.height / 2 - 1.5
                   for o in nrz)
    for p in paths:
        if p:
            assert not any(deep_in_nrz(x, y) for (x, y) in p)   # never through the non-routing zone


def test_equal_length_placement_more_uniform_radius():
    """equal_length_placement puts TPs at a common radius, so pin->connector distances
    are more uniform than the board-filling spread — the real lever for length matching
    (post-hoc meandering is space-limited; measured equalized spread ~0.19 vs ~0.39)."""
    from envs.board import load_te_example, spread_placement, equal_length_placement
    b = load_te_example(num_traces=16, seed=6)
    ccx = b.connector_x + b.connector_w / 2
    ccy = b.connector_y + b.connector_h / 2
    def radius_cv(pl):
        r = [np.hypot(x - ccx, y - ccy) for (x, y) in pl]
        return float(np.std(r) / np.mean(r))
    assert radius_cv(equal_length_placement(b, 16)) < radius_cv(spread_placement(b, 16))


def test_any_angle_shortens_without_reducing_clearance():
    """any_angle_shortcut makes traces shorter (straight any-angle segments) but,
    because each shortcut is verified by exact segment distance, never brings two
    nets closer than they already were, and stays crossing-free."""
    from envs.board import load_te_example, equal_length_placement
    from envs.routing import any_angle_shortcut, min_trace_separation
    b = load_te_example(num_traces=16, seed=6)
    placed = equal_length_placement(b, 16)
    p, L, f = route_all_traces(b, placed)
    plen = lambda P: sum(np.hypot(P[k+1][0]-P[k][0], P[k+1][1]-P[k][1]) for k in range(len(P)-1))
    base_len = sum(plen(x) for x in p if x)
    base_sep = min_trace_separation(p)
    aa = any_angle_shortcut(p, b)
    assert sum(plen(x) for x in aa if x) <= base_len + 1e-6     # never longer
    assert min_trace_separation(aa) >= base_sep - 1e-6          # clearance not reduced
    assert count_crossings(aa) == 0                             # stays planar


def test_router_output_never_crosses():
    """route_all_traces output is always planar: the octilinear router penalizes
    diagonal X-crossings and a final pass drops any net still crossing another, so
    there are zero trace crossings regardless of the (crossing-prone) placement."""
    b = load_te_example(num_traces=16, seed=2)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    placed = []                                   # greedy (crossing-prone) placement
    for t in b.traces:
        d = np.hypot(c[:, 0] - t.start_x, c[:, 1] - t.start_y)
        for i in np.argsort(d):
            if check_tp_spacing(placed, *c[i]):
                placed.append(tuple(c[i])); break
    paths, _, _ = route_all_traces(b, placed)
    assert count_crossings(paths) == 0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
