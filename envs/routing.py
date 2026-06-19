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

# Pad keep-out radius (in cells) that OTHER nets may not enter around each test
# point — endpoint-to-other-trace-body clearance. 3 cells ≈ 4 mm (vs the ~1.3 mm
# trace pitch); routes 20/20 on the planar layout. Larger = more clearance but
# less routing room (tunable; 2–4 all route the planar board).
TP_CLEARANCE_CELLS = 3


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

# Octilinear (8-connected, 45°) moves. Diagonal moves give shorter routes, but
# two paths can cross without sharing a cell by using the two complementary
# diagonals of the same unit square — so diagonal crossings are penalized during
# routing AND any residual crossing is removed afterwards (see _remove_crossings),
# guaranteeing a planar (crossing-free) result.
_NEG_NBR = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _diag_key(r, c, nr, nc):
    """(unit-square, diagonal-type) for a diagonal move. The two diagonals of one
    square share `square` and differ in `type`, so type t crosses type 1-t."""
    return (min(r, nr), min(c, nc)), (0 if (nr - r) == (nc - c) else 1)


def _astar_cost(blocked, cell_cost, diag_present, diag_hist, rows, cols, start, end,
                present_penalty, tp_owner=None, net_id=-1):
    """Octilinear A* on a (row, col) grid. `blocked` hard-blocks cells; `cell_cost`
    adds per-cell congestion; a diagonal move additionally pays for crossing the
    complementary diagonal of its unit square (discouraging X-crossings). If
    `tp_owner` is given, a cell inside another net's test-point keep-out is blocked
    — a net may enter only its own pad's keep-out. Octile heuristic; returns cells
    or None."""
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
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc] or (nr, nc) in seen:
                continue
            if tp_owner is not None and tp_owner[nr, nc] >= 0 and tp_owner[nr, nc] != net_id:
                continue                       # keep clear of other nets' test pads
            if dr and dc:
                if blocked[r, nc] and blocked[nr, c]:
                    continue                       # don't cut the corner between obstacles
                sq, t = _diag_key(r, c, nr, nc)
                step = 1.4142135624
                extra = present_penalty * diag_present.get((sq, 1 - t), 0) + diag_hist.get((sq, t), 0.0)
            else:
                step = 1.0
                extra = 0.0
            ng = g[cur] + step + cell_cost[nr, nc] + extra
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                came[(nr, nc)] = cur
                dx, dy = abs(nr - er), abs(nc - ec)
                h = max(dx, dy) + 0.4142135624 * min(dx, dy)   # octile, admissible
                heapq.heappush(pq, (ng + h, (nr, nc)))
    return None


# ------------------------------------------------------------------
# Public API: negotiated-congestion rip-up-and-reroute
# ------------------------------------------------------------------

def _negotiate(blocked, rows, cols, cells, endpoints, order,
               max_iters, present_penalty, history_inc, tp_owner=None):
    """One negotiated rip-up-and-reroute run, processing nets in `order`.

    Nets may share cells (and complementary diagonals) provisionally at a penalty;
    after each pass a history cost accumulates on over-used cells AND on unit
    squares whose two diagonals are both in use, so both cell-overlaps and
    diagonal X-crossings migrate apart over a few passes. Returns per-net cell
    paths ([(row,col),...] or None)."""
    n = len(cells)
    history = np.zeros((rows, cols))
    present = np.zeros((rows, cols), dtype=np.int32)
    diag_present = {}                 # (square, type) -> count
    diag_hist = {}                    # (square, type) -> history cost
    routes = [None] * n
    net_diags = [None] * n           # per-net list of diagonal keys
    for _ in range(max_iters):
        for i in order:
            if routes[i]:
                for cell in routes[i]:
                    present[cell] -= 1
            if net_diags[i]:
                for key in net_diags[i]:
                    diag_present[key] -= 1
            cost = history + present_penalty * present
            p = _astar_cost(blocked, cost, diag_present, diag_hist, rows, cols,
                            cells[i][0], cells[i][1], present_penalty, tp_owner, i)
            routes[i] = p
            net_diags[i] = []
            if p:
                for cell in p:
                    present[cell] += 1
                for k in range(len(p) - 1):
                    (r, c), (nr, nc) = p[k], p[k + 1]
                    if (nr - r) and (nc - c):
                        key = _diag_key(r, c, nr, nc)
                        diag_present[key] = diag_present.get(key, 0) + 1
                        net_diags[i].append(key)
        cell_conf = [(r, c) for r, c in zip(*np.where(present > 1))
                     if (r, c) not in endpoints]
        squares = {}
        for (sq, t), cnt in diag_present.items():
            if cnt > 0:
                squares.setdefault(sq, set()).add(t)
        diag_conf = [sq for sq, ts in squares.items() if len(ts) == 2]
        if not cell_conf and not diag_conf:
            break
        for cell in cell_conf:
            history[cell] += history_inc
        for sq in diag_conf:
            diag_hist[(sq, 0)] = diag_hist.get((sq, 0), 0.0) + history_inc
            diag_hist[(sq, 1)] = diag_hist.get((sq, 1), 0.0) + history_inc
    return routes


def _remove_crossings(routes, cells, rows, cols, grid):
    """Guarantee a planar result: first drop nets that share a cell, then greedily
    drop nets still involved in a geometric crossing — using the SAME predicate as
    count_crossings (in world coordinates) — most-conflicting net first, until no
    crossings remain. Returns the routes list with offending nets set to None."""
    present = np.zeros((rows, cols), dtype=np.int32)
    for rt in routes:
        if rt:
            for cell in rt:
                present[cell] += 1
    for i in range(len(routes)):
        if routes[i] and any(present[c] > 1 and c not in cells[i] for c in routes[i]):
            for cell in routes[i]:
                present[cell] -= 1
            routes[i] = None
    while True:
        world = [[grid._grid_to_world(c, r) for (r, c) in rt] if rt else None
                 for rt in routes]
        pairs = _crossing_pairs(world)
        if not pairs:
            break
        deg = {}
        for i, j in pairs:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        routes[max(deg, key=lambda k: deg[k])] = None
    return routes


def route_all_traces(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    max_iters: int = 40,
    present_penalty: float = 4.0,
    history_inc: float = 1.0,
    n_starts: int = 6,
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

    # Test-point keep-out: each pad reserves a small disk that OTHER nets may not
    # enter (endpoint-to-other-trace-body and endpoint-to-endpoint clearance); a net
    # may enter only its own pad's disk. Pads are >=13mm apart so disks never
    # overlap. Start/escape cells are exempt so a trace's escape is never boxed in.
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
        step_dir = 1 if sr > crow else -1
        for j in range(6):
            rr = sr + step_dir * j
            if 0 <= rr < rows:
                tp_owner[rr, sc] = -1

    # Multi-start: informed orders first, then deterministic random restarts. Each
    # run is negotiated, then any residual crossing is removed (planar guarantee);
    # keep the routing with the fewest dropped nets, early-exit on a full routing.
    dist = [np.hypot(test_points[i][0] - board.traces[i].start_x,
                     test_points[i][1] - board.traces[i].start_y) for i in range(n)]
    base = list(range(n))
    informed = [base,
                sorted(base, key=lambda i: -dist[i]),   # longest first
                sorted(base, key=lambda i: dist[i])]    # shortest first
    rng = np.random.RandomState(0)
    best_routes, best_fails = None, None
    for k in range(max(1, n_starts)):
        order = informed[k] if k < len(informed) else list(rng.permutation(n))
        routes = _negotiate(blocked, rows, cols, cells, endpoints, order,
                            max_iters, present_penalty, history_inc, tp_owner)
        routes = _remove_crossings(routes, cells, rows, cols, grid)
        fails = sum(1 for rt in routes if rt is None)
        if best_fails is None or fails < best_fails:
            best_fails, best_routes = fails, routes
        if fails == 0:
            break

    routes = best_routes
    paths: List[Optional[List[Tuple[float, float]]]] = []
    lengths: List[float] = []
    failures = 0
    for i in range(n):
        rt = routes[i]
        if rt is None:
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


def route_two_layer(board, test_points, layer_of, **kwargs):
    """Route a multi-layer board. Nets on layer 0 route on the top copper (with all
    obstacles); nets on any other layer route WITHOUT the rect/circ obstacles (the
    connector body / components are top-layer-only, reached by a via to an inner
    layer). Each layer is routed planar and independently by route_all_traces, so
    two traces on DIFFERENT layers may cross (a via), while same-layer crossings are
    still forbidden. Returns (paths_world, lengths, failures, layer_crossings)."""
    import copy
    n = min(len(board.traces), len(test_points))
    paths = [None] * n
    lengths = [float('inf')] * n
    failures = 0
    layer_crossings = {}
    for layer in sorted(set(layer_of[:n])):
        idxs = [i for i in range(n) if layer_of[i] == layer]
        sub = copy.copy(board)
        sub.traces = [board.traces[i] for i in idxs]
        if layer != 0:
            sub.rect_obstacles = []        # components/keep-outs are top-layer only
            sub.circ_obstacles = []
        sp, sl, sf = route_all_traces(sub, [test_points[i] for i in idxs], **kwargs)
        layer_crossings[layer] = count_crossings(sp)
        failures += sf
        for k, i in enumerate(idxs):
            paths[i] = sp[k]
            lengths[i] = sl[k]
    return paths, lengths, failures, layer_crossings


def equalize_lengths(board, paths_world, passes=24, tol=1.0, target_mm=None):
    """Stage 3 — length matching. Meander shorter traces to the longest trace by
    inserting serpentine 'bumps' (a 1-cell perpendicular detour out and back) in
    free cells along each path. Crossing-safe by construction: bumps only ever use
    cells that no obstacle or trace (or earlier bump) occupies, so the result stays
    cell-disjoint and therefore planar.

    Returns (new_paths_world, new_lengths, target_len_mm, n_matched), where
    n_matched is how many routed traces reached the target length (the rest are
    space-limited — that residual is what the soft length-spread reward steers the
    agent to keep small)."""
    grid = RoutingGrid(board)
    rows, cols, res = grid.rows, grid.cols, grid.res
    blocked = grid.obstacle_grid > 0

    # World paths -> dedup'd (row, col) cell paths.
    cell_paths = []
    for p in paths_world:
        if p is None:
            cell_paths.append(None)
            continue
        cp = []
        for (x, y) in p:
            c, r = grid._world_to_grid(x, y)
            if not cp or cp[-1] != (r, c):
                cp.append((r, c))
        cell_paths.append(cp)

    occ = set()
    for cp in cell_paths:
        if cp:
            occ.update(cp)

    def plen(p):
        return sum(np.hypot(p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                   for k in range(len(p) - 1))

    routed = [p for p in cell_paths if p]
    if not routed:
        return paths_world, [float('inf')] * len(paths_world), 0.0, 0
    # target length in cells: a caller-supplied global target (for matching across
    # multiple layers) or the longest routed trace here.
    target = (target_mm / res) if target_mm is not None else max(plen(p) for p in routed)

    for i in range(len(cell_paths)):
        p = cell_paths[i]
        if not p:
            continue
        need = target - plen(p)                 # cells of length still to add
        if need <= tol:
            continue
        added_total = 0.0
        for _ in range(passes):
            if added_total >= need - tol:
                break
            new = [p[0]]
            progressed = False
            endpt, startpt = p[-1], p[0]
            keep = TP_CLEARANCE_CELLS + 1        # clean approach; never wrap a pad
            for k in range(len(p) - 1):
                A, N = p[k], p[k + 1]
                dr, dc = N[0] - A[0], N[1] - A[1]
                near_end = (max(abs(A[0] - endpt[0]), abs(A[1] - endpt[1])) <= keep or
                            max(abs(N[0] - endpt[0]), abs(N[1] - endpt[1])) <= keep or
                            max(abs(A[0] - startpt[0]), abs(A[1] - startpt[1])) <= keep or
                            max(abs(N[0] - startpt[0]), abs(N[1] - startpt[1])) <= keep)
                # Insert a bump only while we still need length (stop at target — each
                # bump adds exactly 2 cells, so we never overshoot by >1 bump) and
                # never within `keep` cells of an endpoint, so a trace never
                # surrounds its own pad (or its start) with its own body.
                if added_total < need - tol and (dr == 0) != (dc == 0) and not near_end:
                    perps = [(1, 0), (-1, 0)] if dr == 0 else [(0, 1), (0, -1)]
                    for pr, pc in perps:
                        B = (A[0] + pr, A[1] + pc)
                        C = (N[0] + pr, N[1] + pc)
                        if (0 <= B[0] < rows and 0 <= B[1] < cols and
                                0 <= C[0] < rows and 0 <= C[1] < cols and
                                not blocked[B] and not blocked[C] and
                                B not in occ and C not in occ and B != new[-1]):
                            new += [B, C]
                            occ.add(B)
                            occ.add(C)
                            added_total += 2.0
                            progressed = True
                            break
                new.append(N)
            p = new
            if not progressed:
                break
        cell_paths[i] = p

    new_paths = [[grid._grid_to_world(c, r) for (r, c) in p] if p else None
                 for p in cell_paths]
    new_lengths = [plen(p) * res + board.traces[i].breakout_length if p else float('inf')
                   for i, p in enumerate(cell_paths)]
    n_matched = sum(1 for p in cell_paths if p and plen(p) >= target - tol)
    return new_paths, new_lengths, target * res, n_matched


def _resample_path(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """Uniformly resample a path to at most n points (keeps endpoints).
    Uniform sampling, unlike strided slicing, never drops the final point."""
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


def _crossing_pairs(paths):
    """Set of (i, j) net-index pairs whose routed polylines properly cross. Exact
    O(#segments^2) check — the shared, safety-critical primitive for the planar
    guarantee (used by count_crossings and the router's crossing-removal pass)."""
    segs = []
    for idx, p in enumerate(paths):
        if p:
            for k in range(len(p) - 1):
                segs.append((idx, p[k], p[k + 1]))
    pairs = set()
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
                pairs.add((min(ia, ib), max(ia, ib)))
    return pairs


def count_crossings(paths) -> int:
    """Number of trace pairs that properly cross — a valid single-layer routing
    has zero. Defensive check used in validation and tests."""
    return len(_crossing_pairs(paths))


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

    # Test-point (pad) to other-trace-body clearance: each pad should stay clear of
    # every trace except its own (the pad keep-out the router enforces).
    tp2trace_min = float('inf')
    for i, pi in enumerate(paths):
        if not pi:
            continue
        tp = np.asarray(pi[-1])
        for j, pj in enumerate(paths):
            if pj is None or j == i:
                continue
            arr = np.asarray(pj)
            tp2trace_min = min(tp2trace_min,
                               float(np.hypot(arr[:, 0] - tp[0], arr[:, 1] - tp[1]).min()))

    return {"violations": violations, "trace_to_trace_min": t2t_min,
            "trace_to_edge_min": t2e_min, "trace_to_obstacle_min": t2o_min,
            "crossings": crossings, "tp_to_trace_min": tp2trace_min,
            "all_valid": len(violations) == 0 and crossings == 0}