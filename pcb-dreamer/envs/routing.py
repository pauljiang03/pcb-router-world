"""
A* routing on a cell-discretized PCB grid with rip-up-and-retry.

Cell size = TRACE_WIDTH + TRACE_TO_TRACE_MIN guarantees clearance between
adjacent traces.  Test-point clearance zones prevent traces from passing
too close to other traces' test points.  Diagonal crossings are tracked
and forbidden.  First step from each starting point must be cardinal.

Rip-up-and-retry: if routing in the default order leaves failures, the
router tries alternative orderings (reverse, distance-sorted, failed-
first, random shuffles) and keeps the result with fewest failures.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Set, Dict, FrozenSet
from envs.board import (
    BoardSpec, TRACE_WIDTH, TRACE_TO_TRACE_MIN,
    TRACE_TO_EDGE_MIN, TRACE_TO_UPTH_MIN, TRACE_TO_TABPAD_MIN,
)

CELL_SIZE = TRACE_WIDTH + TRACE_TO_TRACE_MIN  # 1.3286 mm

# How many cells around each test point are blocked for other traces.
# 2 cells ≈ 2.66mm clearance from TP center.
TP_CLEARANCE_CELLS = 2


class RoutingGrid:

    def __init__(self, board: BoardSpec):
        self.board = board
        self.res = CELL_SIZE
        self.cols = int(np.ceil(board.width / self.res))
        self.rows = int(np.ceil(board.height / self.res))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        self._rasterize_edge_clearance()
        self._rasterize_obstacles()
        self.obstacle_grid = self.grid.copy()

        self._start_cells: Dict[int, Tuple[int, int]] = {}
        for trace in board.traces:
            c, r = self._world_to_grid(trace.start_x, trace.start_y)
            self._start_cells[trace.index] = (c, r)

        self._remaining_traces: Set[int] = set(t.index for t in board.traces)
        self._occupied: Set[Tuple[int, int]] = set()
        self._blocked_diags: Set[FrozenSet] = set()

        # TP clearance zones: trace_index -> set of (col, row) cells blocked
        # around that trace's test point.
        self._tp_zones: Dict[int, Set[Tuple[int, int]]] = {}

        self._clear_remaining_starts()

    # ---- coordinates ----

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        c = int(round((x - self.board.x_min) / self.res - 0.5))
        r = int(round((y - self.board.y_min) / self.res - 0.5))
        return int(np.clip(c, 0, self.cols - 1)), int(np.clip(r, 0, self.rows - 1))

    def _grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return (self.board.x_min + (col + 0.5) * self.res,
                self.board.y_min + (row + 0.5) * self.res)

    # ---- initial rasterization ----

    def _rasterize_edge_clearance(self):
        n = max(1, int(np.ceil((TRACE_TO_EDGE_MIN + TRACE_WIDTH / 2) / self.res)))
        self.grid[:n, :] = 1
        self.grid[-n:, :] = 1
        self.grid[:, :n] = 1
        self.grid[:, -n:] = 1

    def _rasterize_obstacles(self):
        hw = TRACE_WIDTH / 2
        for obs in self.board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            b = obs.clearance + hw
            c0, r0 = self._world_to_grid(xn - b, yn - b)
            c1, r1 = self._world_to_grid(xx + b, yx + b)
            self.grid[max(0, r0):min(self.rows, r1 + 1),
                      max(0, c0):min(self.cols, c1 + 1)] = 1
        for obs in self.board.circ_obstacles:
            b = obs.radius + obs.clearance + hw
            c0, r0 = self._world_to_grid(obs.cx - b, obs.cy - b)
            c1, r1 = self._world_to_grid(obs.cx + b, obs.cy + b)
            for r in range(max(0, r0), min(self.rows, r1 + 1)):
                for c in range(max(0, c0), min(self.cols, c1 + 1)):
                    wx, wy = self._grid_to_world(c, r)
                    if np.hypot(wx - obs.cx, wy - obs.cy) < b:
                        self.grid[r, c] = 1

    # ---- test-point clearance zones ----

    def set_test_points(self, trace_indices: List[int],
                        test_points: List[Tuple[float, float]]):
        """Block cells around each test point so other traces avoid them."""
        rad = TP_CLEARANCE_CELLS
        for ti, (tx, ty) in zip(trace_indices, test_points):
            tc, tr = self._world_to_grid(tx, ty)
            zone: Set[Tuple[int, int]] = set()
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if dr * dr + dc * dc <= rad * rad:
                            zone.add((nc, nr))
            self._tp_zones[ti] = zone
            # Block the zone on the grid
            for (zc, zr) in zone:
                self.grid[zr, zc] = 1
        # Re-clear starting points (they might have been blocked by TP zones)
        self._clear_remaining_starts()

    # ---- starting-point management ----

    def _clear_remaining_starts(self):
        for idx in self._remaining_traces:
            c, r = self._start_cells[idx]
            if (c, r) not in self._occupied:
                self.grid[r, c] = 0

    # ---- trace blocking ----

    def rasterize_trace_path(self, path: List[Tuple[int, int]],
                             routed_trace_index: int):
        for col, row in path:
            self.grid[row, col] = 1
            self._occupied.add((col, row))
        for i in range(len(path) - 1):
            c1, r1 = path[i]
            c2, r2 = path[i + 1]
            if (c2 - c1) != 0 and (r2 - r1) != 0:
                self._blocked_diags.add(frozenset(((c2, r1), (c1, r2))))
        self._remaining_traces.discard(routed_trace_index)
        self._clear_remaining_starts()

    # ---- A* with congestion penalty ----

    _NBR = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
    _SQRT2 = float(np.sqrt(2))

    # Penalty per occupied neighbor — steers traces away from congested areas
    CONGESTION_WEIGHT = 0.6

    def _congestion_cost(self, col: int, row: int) -> float:
        """Extra cost for cells near occupied traces. Spreads routes apart."""
        penalty = 0.0
        for dc, dr in self._NBR:
            nc, nr = col + dc, row + dr
            if 0 <= nc < self.cols and 0 <= nr < self.rows:
                if (nc, nr) in self._occupied:
                    penalty += self.CONGESTION_WEIGHT
        return penalty

    def find_path(self, sx: float, sy: float, ex: float, ey: float,
                  trace_index: int = -1
                  ) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        sc, sr = self._world_to_grid(sx, sy)
        ec, er = self._world_to_grid(ex, ey)

        # Temporarily block other remaining starting cells
        saved: List[Tuple[int, int, int]] = []
        for idx in self._remaining_traces:
            if idx == trace_index:
                continue
            oc, orow = self._start_cells[idx]
            if (oc, orow) != (sc, sr) and (oc, orow) != (ec, er):
                saved.append((oc, orow, self.grid[orow, oc]))
                self.grid[orow, oc] = 1

        # Temporarily clear THIS trace's TP zone so it can reach its own TP
        tp_zone_cleared: List[Tuple[int, int, int]] = []
        if trace_index in self._tp_zones:
            for (zc, zr) in self._tp_zones[trace_index]:
                if (zc, zr) not in self._occupied:
                    tp_zone_cleared.append((zc, zr, self.grid[zr, zc]))
                    self.grid[zr, zc] = 0

        # Clear start/end if not occupied
        os_ = self.grid[sr, sc]
        oe_ = self.grid[er, ec]
        if (sc, sr) not in self._occupied:
            self.grid[sr, sc] = 0
        if (ec, er) not in self._occupied:
            self.grid[er, ec] = 0

        # A*
        heap: list = [(0.0, sc, sr)]
        came: dict = {}
        g: dict = {(sc, sr): 0.0}
        vis: Set[Tuple[int, int]] = set()
        ok = False

        while heap:
            _, cc, cr = heapq.heappop(heap)
            if (cc, cr) in vis:
                continue
            vis.add((cc, cr))
            if cc == ec and cr == er:
                ok = True
                break
            for dc, dr in self._NBR:
                nc, nr = cc + dc, cr + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    if self.grid[nr, nc] == 0 and (nc, nr) not in vis:
                        if cc == sc and cr == sr and dc != 0 and dr != 0:
                            continue  # cardinal first step
                        if dc != 0 and dr != 0:
                            if frozenset(((cc, cr), (nc, nr))) in self._blocked_diags:
                                continue
                        cost = self._SQRT2 if (dc and dr) else 1.0
                        cost += self._congestion_cost(nc, nr)
                        tg = g[(cc, cr)] + cost
                        if tg < g.get((nc, nr), float('inf')):
                            g[(nc, nr)] = tg
                            came[(nc, nr)] = (cc, cr)
                            heapq.heappush(heap,
                                           (tg + np.hypot(nc - ec, nr - er), nc, nr))

        # Restore grid
        self.grid[sr, sc] = os_
        self.grid[er, ec] = oe_
        for zc, zr, zv in tp_zone_cleared:
            self.grid[zr, zc] = zv
        for oc, orow, ov in saved:
            self.grid[orow, oc] = ov

        if not ok:
            return None

        path = [(ec, er)]
        c, r = ec, er
        while (c, r) in came:
            c, r = came[(c, r)]
            path.append((c, r))
        path.reverse()
        length = sum(np.hypot(path[i + 1][0] - path[i][0],
                              path[i + 1][1] - path[i][1])
                     for i in range(len(path) - 1)) * self.res
        return path, length

    def path_to_world(self, path):
        return [self._grid_to_world(c, r) for c, r in path]


# ------------------------------------------------------------------
# Single-order routing (used internally by rip-up-and-retry)
# ------------------------------------------------------------------

def _route_in_order(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    order: List[int],
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """Route traces in `order` on a fresh grid.  Returns (paths, lengths, failures)."""
    grid = RoutingGrid(board)

    # Set up TP clearance zones for ALL test points
    trace_indices = [board.traces[i].index for i in range(min(len(board.traces), len(test_points)))]
    grid.set_test_points(trace_indices, test_points[:len(trace_indices)])

    # Route in the given order
    result_paths: Dict[int, Optional[List[Tuple[float, float]]]] = {}
    result_lengths: Dict[int, float] = {}
    failures = 0

    for i in order:
        if i >= len(board.traces) or i >= len(test_points):
            continue
        trace = board.traces[i]
        tp_x, tp_y = test_points[i]
        result = grid.find_path(trace.start_x, trace.start_y, tp_x, tp_y,
                                trace_index=trace.index)
        if result is None:
            result_paths[i] = None
            result_lengths[i] = float('inf')
            failures += 1
            grid._remaining_traces.discard(trace.index)
        else:
            grid_path, path_length = result
            world_path = grid.path_to_world(grid_path)
            result_paths[i] = world_path
            result_lengths[i] = path_length + trace.breakout_length
            grid.rasterize_trace_path(grid_path, trace.index)

    # Reassemble in original trace order
    n = min(len(board.traces), len(test_points))
    paths = [result_paths.get(i) for i in range(n)]
    lengths = [result_lengths.get(i, float('inf')) for i in range(n)]
    return paths, lengths, failures


# ------------------------------------------------------------------
# Public API: rip-up-and-retry routing
# ------------------------------------------------------------------

def route_all_traces(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    max_retries: int = 15,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """
    Route all traces with rip-up-and-retry.

    Tries multiple routing orders and keeps the result with fewest failures:
      1. Original order
      2. Reverse
      3. Shortest straight-line distance first
      4. Longest distance first
      5. Failed-traces-first (from best attempt so far)
      6. Random shuffles
    """
    n = min(len(board.traces), len(test_points))
    if n == 0:
        return [], [], 0

    # Compute straight-line distances for ordering heuristics
    dists = []
    for i in range(n):
        t = board.traces[i]
        dx = test_points[i][0] - t.start_x
        dy = test_points[i][1] - t.start_y
        dists.append(np.hypot(dx, dy))

    # Generate candidate orders
    orders: List[List[int]] = []
    base = list(range(n))
    orders.append(base)                                         # original
    orders.append(base[::-1])                                   # reverse
    orders.append(sorted(base, key=lambda i: dists[i]))         # shortest first
    orders.append(sorted(base, key=lambda i: -dists[i]))        # longest first

    # Route with best order so far
    best_paths, best_lengths, best_failures = _route_in_order(board, test_points, orders[0])

    for order in orders[1:]:
        if best_failures == 0:
            break
        paths, lengths, failures = _route_in_order(board, test_points, order)
        if failures < best_failures:
            best_paths, best_lengths, best_failures = paths, lengths, failures

    # Failed-first retry: move previously-failed traces to the front
    if best_failures > 0:
        failed = [i for i in range(n) if best_paths[i] is None]
        succeeded = [i for i in range(n) if best_paths[i] is not None]
        order = failed + succeeded
        paths, lengths, failures = _route_in_order(board, test_points, order)
        if failures < best_failures:
            best_paths, best_lengths, best_failures = paths, lengths, failures

    # Random shuffles
    rng = np.random.RandomState(42)
    for _ in range(max_retries):
        if best_failures == 0:
            break
        order = list(rng.permutation(n))
        paths, lengths, failures = _route_in_order(board, test_points, order)
        if failures < best_failures:
            best_paths, best_lengths, best_failures = paths, lengths, failures
            # Try failed-first on this new best
            failed = [i for i in range(n) if best_paths[i] is None]
            succeeded = [i for i in range(n) if best_paths[i] is not None]
            order2 = failed + succeeded
            paths2, lengths2, failures2 = _route_in_order(board, test_points, order2)
            if failures2 < best_failures:
                best_paths, best_lengths, best_failures = paths2, lengths2, failures2

    return best_paths, best_lengths, best_failures


def validate_routing_constraints(
    board: BoardSpec,
    paths: List[Optional[List[Tuple[float, float]]]],
) -> dict:
    """Check all hard constraints on routed traces."""
    violations = []
    t2t_min = float('inf')
    t2e_min = float('inf')
    t2o_min = float('inf')
    hw = TRACE_WIDTH / 2

    vp, vi = [], []
    for i, p in enumerate(paths):
        if p is not None:
            vp.append(np.array(p))
            vi.append(i)

    for idx, pts in zip(vi, vp):
        de = min((pts[:, 0] - board.x_min).min() - hw,
                 (board.x_max - pts[:, 0]).min() - hw,
                 (pts[:, 1] - board.y_min).min() - hw,
                 (board.y_max - pts[:, 1]).min() - hw)
        t2e_min = min(t2e_min, de)
        if de < TRACE_TO_EDGE_MIN - CELL_SIZE:
            violations.append(("trace_to_edge", idx, None,
                               f"Trace {idx}: edge dist {de:.3f}mm"))

    for idx, pts in zip(vi, vp):
        for obs in board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            dx = np.maximum(np.maximum(xn - pts[:, 0], 0), pts[:, 0] - xx)
            dy = np.maximum(np.maximum(yn - pts[:, 1], 0), pts[:, 1] - yx)
            d = np.where((dx > 0) | (dy > 0), np.hypot(dx, dy), 0.0) - hw
            dm = float(d.min())
            t2o_min = min(t2o_min, dm)
            if dm < obs.clearance - CELL_SIZE:
                violations.append(("trace_to_obstacle", idx, obs.name,
                                   f"Trace {idx}: {obs.name} dist {dm:.3f}mm"))
        for obs in board.circ_obstacles:
            d = np.hypot(pts[:, 0] - obs.cx, pts[:, 1] - obs.cy) - obs.radius - hw
            dm = float(d.min())
            t2o_min = min(t2o_min, dm)
            if dm < obs.clearance - CELL_SIZE:
                violations.append(("trace_to_obstacle", idx, obs.name,
                                   f"Trace {idx}: {obs.name} dist {dm:.3f}mm"))

    for a in range(len(vp)):
        for b in range(a + 1, len(vp)):
            sa = vp[a][::max(1, len(vp[a]) // 200)]
            sb = vp[b][::max(1, len(vp[b]) // 200)]
            d = np.sqrt(((sa[:, None, :] - sb[None, :, :]) ** 2).sum(2))
            mcc = float(d.min())
            mee = mcc - TRACE_WIDTH
            t2t_min = min(t2t_min, mee)
            if mee < TRACE_TO_TRACE_MIN - CELL_SIZE:
                violations.append(("trace_to_trace", vi[a], vi[b],
                                   f"Traces {vi[a]},{vi[b]}: edge dist {mee:.3f}mm"))

    return {"violations": violations, "trace_to_trace_min": t2t_min,
            "trace_to_edge_min": t2e_min, "trace_to_obstacle_min": t2o_min,
            "all_valid": len(violations) == 0}