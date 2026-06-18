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
# Negotiated-congestion router internals
# ------------------------------------------------------------------

# Rectilinear (4-connected) moves ONLY. On a grid, two axis-aligned cell-disjoint
# paths cannot cross — so a conflict-free routing is automatically planar (no
# trace crossings) BY CONSTRUCTION. Diagonal moves are intentionally excluded
# because they permit X-crossings between two paths that share no cell.
_NEG_NBR = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _astar_cost(blocked, cost, rows, cols, start, end):
    """Rectilinear A* on a (row, col) grid. `blocked` is a hard boolean mask;
    `cost` adds a per-cell congestion penalty to each entered cell. Returns a
    list of (row, col) cells from start to end, or None if unreachable."""
    g = {start: 0.0}
    came = {}
    pq = [(0.0, start)]
    seen = set()
    er, ec = end
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == end:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        r, c = cur
        for dr, dc in _NEG_NBR:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr, nc] and (nr, nc) not in seen:
                step = 1.4142135624 if (dr and dc) else 1.0
                ng = g[cur] + step + cost[nr, nc]
                if ng < g.get((nr, nc), 1e18):
                    g[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    # Manhattan heuristic — tight & admissible for 4-connected moves.
                    heapq.heappush(pq, (ng + abs(nr - er) + abs(nc - ec), (nr, nc)))
    return None


# ------------------------------------------------------------------
# Public API: negotiated-congestion rip-up-and-reroute
# ------------------------------------------------------------------

def _negotiate(blocked, rows, cols, cells, endpoints, order,
               max_iters, present_penalty, history_inc):
    """One negotiated rip-up-and-reroute run, processing nets in `order`.

    Nets may share cells provisionally at a congestion penalty; after each pass
    a per-cell `history` cost accumulates on over-used cells, so contention
    resolves over a few passes. Returns per-net cell paths ([(row,col),...] or
    None)."""
    n = len(cells)
    history = np.zeros((rows, cols))
    present = np.zeros((rows, cols), dtype=np.int32)
    routes = [None] * n
    for _ in range(max_iters):
        for i in order:
            if routes[i]:
                for cell in routes[i]:
                    present[cell] -= 1
            cost = history + present_penalty * present
            routes[i] = _astar_cost(blocked, cost, rows, cols, cells[i][0], cells[i][1])
            if routes[i]:
                for cell in routes[i]:
                    present[cell] += 1
        shared = [(r, c) for r, c in zip(*np.where(present > 1))
                  if (r, c) not in endpoints]
        if not shared:
            break
        for cell in shared:
            history[cell] += history_inc
    return routes


def _score_routes(routes, cells, rows, cols):
    """Return (failures, length_in_cells, present_grid). A net fails if it has no
    path or shares a non-endpoint cell with another net."""
    present = np.zeros((rows, cols), dtype=np.int32)
    for rt in routes:
        if rt:
            for cell in rt:
                present[cell] += 1
    fails = 0
    length_cells = 0.0
    for i in range(len(routes)):
        rt = routes[i]
        if rt is None or any(present[c] > 1 and c not in cells[i] for c in rt):
            fails += 1
        elif rt:
            length_cells += sum(np.hypot(rt[k + 1][0] - rt[k][0], rt[k + 1][1] - rt[k][1])
                                for k in range(len(rt) - 1))
    return fails, length_cells, present


def route_all_traces(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    max_iters: int = 40,
    present_penalty: float = 3.0,
    history_inc: float = 1.0,
    n_starts: int = 8,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """
    Route all traces with negotiated-congestion rip-up-and-reroute + multi-start.

    A feasible routing almost always exists on these boards (e.g. a radial fan on
    a central connector); the difficulty is that one-shot sequential A* — route a
    net, hard-block its corridor, move on — boxes later nets out of solutions that
    demonstrably exist. Two mechanisms recover them:

      1. Negotiated congestion: nets share cells provisionally at a penalty, then
         iteratively rip up and reroute while a per-cell history cost accumulates
         on contested cells, so contention resolves and routes stay short.
      2. Multi-start: routing order has a large effect on the outcome (the same
         placement can route with 1 or 6 failures depending on order), so we try
         a few informed orders (identity, longest-first, shortest-first) then
         deterministic random restarts, keeping the best (fewest failures, then
         shortest), with early-exit once a conflict-free routing is found.

    Returns (paths_world, lengths, failures), matching the original API:
      paths_world[i]: list of (x, y) world points, or None if net i failed.
      lengths[i]:     routed length + breakout, or inf if failed.
      failures:       number of nets with no legal (conflict-free) route.
    """
    n = min(len(board.traces), len(test_points))
    if n == 0:
        return [], [], 0

    grid = RoutingGrid(board)
    rows, cols, res = grid.rows, grid.cols, grid.res
    blocked = grid.obstacle_grid > 0           # hard obstacles + board edge

    # Map start / test-point to (row, col) cells; clear them so they're routable.
    cells = []
    for i in range(n):
        t = board.traces[i]
        sc, sr = grid._world_to_grid(t.start_x, t.start_y)
        ec, er = grid._world_to_grid(*test_points[i])
        cells.append(((sr, sc), (er, ec)))
        blocked[sr, sc] = False
        blocked[er, ec] = False

    # Carve a short escape stub outward from each start. The connector keep-out
    # raster over-blocks the cluster, so without this some starts are boxed in
    # (every A* would return no-path). Each start escapes away from the cluster
    # center along its row direction (lower row downward, upper row upward).
    ccx = board.connector_x + board.connector_w / 2.0
    ccy = board.connector_y + board.connector_h / 2.0
    crow = grid._world_to_grid(ccx, ccy)[1]
    for (sr, sc), _e in cells:
        step_dir = 1 if sr > crow else -1
        for j in range(5):
            rr = sr + step_dir * j
            if 0 <= rr < rows:
                blocked[rr, sc] = False

    endpoints = set()
    for s, e in cells:
        endpoints.add(s)
        endpoints.add(e)

    # Multi-start: informed orders first, then deterministic random restarts.
    dist = [np.hypot(test_points[i][0] - board.traces[i].start_x,
                     test_points[i][1] - board.traces[i].start_y) for i in range(n)]
    base = list(range(n))
    informed = [base,
                sorted(base, key=lambda i: -dist[i]),   # longest first
                sorted(base, key=lambda i: dist[i])]    # shortest first
    rng = np.random.RandomState(0)
    best_routes, best_key = None, None
    for k in range(max(1, n_starts)):
        order = informed[k] if k < len(informed) else list(rng.permutation(n))
        routes = _negotiate(blocked, rows, cols, cells, endpoints, order,
                            max_iters, present_penalty, history_inc)
        fails, length_cells, _ = _score_routes(routes, cells, rows, cols)
        key = (fails, length_cells)
        if best_key is None or key < best_key:
            best_key, best_routes = key, routes
        if fails == 0:
            break

    routes = best_routes
    _, _, present = _score_routes(routes, cells, rows, cols)

    paths: List[Optional[List[Tuple[float, float]]]] = []
    lengths: List[float] = []
    failures = 0
    for i in range(n):
        rt = routes[i]
        s, e = cells[i]
        illegal = rt is None or any(
            present[cell] > 1 and cell not in (s, e) for cell in rt
        )
        if illegal:
            paths.append(None)
            lengths.append(float('inf'))
            failures += 1
        else:
            world = [grid._grid_to_world(c, r) for (r, c) in rt]
            plen = sum(np.hypot(rt[k + 1][0] - rt[k][0], rt[k + 1][1] - rt[k][1])
                       for k in range(len(rt) - 1)) * res
            paths.append(world)
            lengths.append(plen + board.traces[i].breakout_length)
    return paths, lengths, failures


def _resample_path(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """Uniformly resample a path to at most n points (keeps endpoints).
    Uniform sampling, unlike strided slicing, never drops the final point."""
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


def count_crossings(paths) -> int:
    """Count pairs of traces whose routed polylines properly cross. Two traces
    crossing is illegal on a single layer; a valid routing has zero crossings.
    (The rectilinear router guarantees zero among conflict-free nets, so this is
    a defensive check used in validation and tests.)"""
    segs = []
    for idx, p in enumerate(paths):
        if p:
            for k in range(len(p) - 1):
                segs.append((idx, p[k], p[k + 1]))
    crossing_pairs = set()
    for a in range(len(segs)):
        ia, A, B = segs[a]
        for b in range(a + 1, len(segs)):
            ib, C, D = segs[b]
            if ia == ib:
                continue
            d1 = _ccw(C[0], C[1], D[0], D[1], A[0], A[1])
            d2 = _ccw(C[0], C[1], D[0], D[1], B[0], B[1])
            d3 = _ccw(A[0], A[1], B[0], B[1], C[0], C[1])
            d4 = _ccw(A[0], A[1], B[0], B[1], D[0], D[1])
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                crossing_pairs.add((min(ia, ib), max(ia, ib)))
    return len(crossing_pairs)


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
            sa = _resample_path(vp[a])
            sb = _resample_path(vp[b])
            d = np.sqrt(((sa[:, None, :] - sb[None, :, :]) ** 2).sum(2))
            mcc = float(d.min())
            mee = mcc - TRACE_WIDTH
            t2t_min = min(t2t_min, mee)
            if mee < TRACE_TO_TRACE_MIN - CELL_SIZE:
                violations.append(("trace_to_trace", vi[a], vi[b],
                                   f"Traces {vi[a]},{vi[b]}: edge dist {mee:.3f}mm"))

    crossings = count_crossings(paths)
    return {"violations": violations, "trace_to_trace_min": t2t_min,
            "trace_to_edge_min": t2e_min, "trace_to_obstacle_min": t2o_min,
            "crossings": crossings,
            "all_valid": len(violations) == 0 and crossings == 0}