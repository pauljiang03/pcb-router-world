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


def test_router_output_never_crosses():
    """The rectilinear router is planar by construction: cell-disjoint axis-aligned
    paths cannot cross, and conflicting nets are dropped — so route_all_traces
    output has zero trace crossings regardless of the (crossing-prone) placement."""
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
