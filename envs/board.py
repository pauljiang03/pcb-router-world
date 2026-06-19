"""
Board representation for SI test fixture routing.

Encodes board geometry, obstacles, starting points, and constraints.
Provides candidate grid generation and constraint checking.

Board data comes from TE Connectivity AutoLayout_Example01 Excel/PowerPoint.
Starting-point spacing is adjusted to exactly satisfy the minimum trace-to-
trace clearance constraint (original 0.9mm spacing violates it; we use
TRACE_TO_TRACE_MIN + TRACE_WIDTH ≈ 1.3286mm).

When a seed is provided, the connector cluster (NRZ, UPTHs, tab pads,
starting traces) is shifted to a random position on the board while
maintaining minimum margin from all edges.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Obstacle:
    """Rectangular obstacle or keep-out zone."""
    cx: float  # center x
    cy: float  # center y
    width: float
    height: float
    clearance: float  # minimum trace-to-obstacle distance (edge-to-edge)
    name: str = ""

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        hw, hh = self.width / 2, self.height / 2
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)


@dataclass
class CircularObstacle:
    """Circular obstacle (UPTH, via, etc.)."""
    cx: float
    cy: float
    radius: float
    clearance: float  # edge-to-edge
    name: str = ""


@dataclass
class TraceSpec:
    """Specification for one trace to be routed."""
    start_x: float
    start_y: float
    breakout_length: float
    index: int


@dataclass
class BoardSpec:
    """Complete board specification loaded from data."""
    origin_x: float
    origin_y: float
    width: float
    height: float

    rect_obstacles: List[Obstacle] = field(default_factory=list)
    circ_obstacles: List[CircularObstacle] = field(default_factory=list)
    traces: List[TraceSpec] = field(default_factory=list)

    # Connector outline (simplified as rectangle)
    connector_x: float = 0.0
    connector_y: float = 0.0
    connector_w: float = 0.0
    connector_h: float = 0.0

    @property
    def x_min(self):
        return self.origin_x

    @property
    def y_min(self):
        return self.origin_y

    @property
    def x_max(self):
        return self.origin_x + self.width

    @property
    def y_max(self):
        return self.origin_y + self.height


# ---------------------------------------------------------------------------
# Fixed constraints (from TE AutoLayout Example01 Excel)
# ---------------------------------------------------------------------------
TRACE_WIDTH = 0.2286        # mm
TRACE_TO_EDGE_MIN = 0.26    # mm, edge-to-edge
TRACE_TO_TRACE_MIN = 1.1    # mm, edge-to-edge
TRACE_TO_UPTH_MIN = 0.7     # mm, edge-to-edge
TRACE_TO_TABPAD_MIN = 0.7   # mm, edge-to-edge
TP_TO_TP_MIN = 13.0         # mm, center-to-center
TP_TO_EDGE_MIN = 14.0       # mm, center-to-edge (from PCB Routine Material)
TP_TO_CONNECTOR_MIN = 3.0   # mm, center-to-edge (from PCB Routine Material)

# Minimum center-to-center trace spacing that satisfies edge-to-edge clearance.
TRACE_MIN_CENTER_TO_CENTER = TRACE_TO_TRACE_MIN + TRACE_WIDTH  # 1.3286mm

# Fixed action-space size so it stays constant across board geometries.
MAX_CANDIDATES = 200

# Central-connector board sizing.
BOARD_SIZE = 160.0          # square board edge, mm (enlarge if routing fails)
N_PER_ROW = 10              # starts per row; 2 rows -> 20 traces


def _respaced_x(original_x: List[float], min_spacing: float) -> List[float]:
    """
    Re-space a list of x-coordinates so the gaps between consecutive
    points are at least `min_spacing`, preserving the group center.

    If the original spacing already satisfies the constraint, positions
    are returned unchanged.
    """
    n = len(original_x)
    if n <= 1:
        return list(original_x)
    center = np.mean(original_x)
    new_x = [center + (i - (n - 1) / 2) * min_spacing for i in range(n)]
    return new_x


def load_te_example(num_traces: int = 20, seed: int = None,
                    board_size: float = BOARD_SIZE) -> BoardSpec:
    """
    Central-connector board for SI test-point placement.

    A connector cluster (non-routing zone, UPTHs, tab pads, and 2 x N_PER_ROW
    start points in two rows) is placed at the CENTER of a square board. The two
    rows straddle the non-routing zone: the lower row escapes downward and the
    upper row escapes upward, so traces fan into both halves of the board rather
    than all routing one direction. This keeps routes short and avoids the
    single-channel congestion of an edge-mounted connector.

    Args:
        num_traces: traces to include (1..2*N_PER_ROW = 20). Lower row first,
                    then upper row.
        seed:       if given, the cluster is jittered around the board center
                    (staying central and routable) so each seed differs.
        board_size: square board edge length in mm.
    """
    board = BoardSpec(origin_x=0.0, origin_y=0.0,
                      width=board_size, height=board_size)

    cx0 = board.x_min + board.width / 2.0
    cy0 = board.y_min + board.height / 2.0

    min_sp = TRACE_MIN_CENTER_TO_CENTER
    row_span = (N_PER_ROW - 1) * min_sp          # x-extent spanned by one row

    nrz_w = row_span + 4.0                        # non-routing zone spans the row
    nrz_h = 6.64
    row_gap = nrz_h + 0.2                         # vertical distance between rows

    # Jitter the cluster around the center for different seeds (kept central).
    if seed is not None:
        rng = np.random.RandomState(seed)
        jitter = min(20.0, board.width / 2 - (TP_TO_EDGE_MIN + max(nrz_w, row_gap)))
        ccx = cx0 + rng.uniform(-jitter, jitter)
        ccy = cy0 + rng.uniform(-jitter, jitter)
    else:
        ccx, ccy = cx0, cy0

    # Non-routing zone centered on the cluster.
    board.rect_obstacles.append(Obstacle(
        cx=ccx, cy=ccy, width=nrz_w, height=nrz_h,
        clearance=TRACE_TO_EDGE_MIN, name="non_routing_zone",
    ))

    # UPTHs inside the NRZ region.
    board.circ_obstacles.append(CircularObstacle(
        cx=ccx - nrz_w * 0.25, cy=ccy, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_1"))
    board.circ_obstacles.append(CircularObstacle(
        cx=ccx + nrz_w * 0.25, cy=ccy, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_2"))

    # Tab pads at the two ends of the cluster.
    for sgn, nm in ((-1, "tab_pad_1"), (1, "tab_pad_2")):
        board.rect_obstacles.append(Obstacle(
            cx=ccx + sgn * (row_span / 2 + 1.2), cy=ccy,
            width=1.526, height=1.216,
            clearance=TRACE_TO_TABPAD_MIN, name=nm))

    # Connector outline encompassing the cluster.
    conn_w = row_span + 6.0
    conn_h = row_gap + 6.0
    board.connector_x = ccx - conn_w / 2
    board.connector_y = ccy - conn_h / 2
    board.connector_w = conn_w
    board.connector_h = conn_h

    # Start points: two rows straddling the NRZ, centered on the cluster.
    start_xs = [ccx + (i - (N_PER_ROW - 1) / 2.0) * min_sp
                for i in range(N_PER_ROW)]
    lower_y = ccy - nrz_h / 2 - 0.1               # escapes downward
    upper_y = ccy + nrz_h / 2 + 0.1               # escapes upward
    breakout = 0.8626

    all_traces = []
    for i, x in enumerate(start_xs):              # lower row: 0 .. N-1
        all_traces.append(TraceSpec(start_x=x, start_y=lower_y,
                                    breakout_length=breakout, index=i))
    for i, x in enumerate(start_xs):              # upper row: N .. 2N-1
        all_traces.append(TraceSpec(start_x=x, start_y=upper_y,
                                    breakout_length=breakout, index=N_PER_ROW + i))

    board.traces = all_traces[:num_traces]
    return board


def load_edge_board(num_traces: int = 20, board_w: float = 180.0,
                    board_h: float = 180.0, seed: int = None) -> BoardSpec:
    """Edge-connector board: a single row of `num_traces` pins LOW on the board,
    all escaping UPWARD, for routing every trace to test points in the upper region
    (a planar fan to the top). Exploratory layout — see docs/ROUTING.md."""
    board = BoardSpec(origin_x=0.0, origin_y=0.0, width=board_w, height=board_h)
    cx = board_w / 2.0
    if seed is not None:
        cx += np.random.RandomState(seed).uniform(-board_w * 0.1, board_w * 0.1)
    row_y = board_h * 0.12                       # low on the board
    sp = TRACE_MIN_CENTER_TO_CENTER
    xs = [cx + (i - (num_traces - 1) / 2) * sp for i in range(num_traces)]
    span = (num_traces - 1) * sp
    # Non-routing zone + UPTH below the pin row, so traces must escape upward.
    board.rect_obstacles.append(Obstacle(
        cx=cx, cy=row_y - 6.0, width=span + 4.0, height=7.0,
        clearance=TRACE_TO_EDGE_MIN, name="non_routing_zone"))
    board.circ_obstacles.append(CircularObstacle(
        cx=cx, cy=row_y - 6.0, radius=0.95, clearance=TRACE_TO_UPTH_MIN, name="UPTH_1"))
    board.connector_x = cx - span / 2 - 3.0
    board.connector_y = row_y - 10.0
    board.connector_w = span + 6.0
    board.connector_h = 13.0
    board.traces = [TraceSpec(start_x=x, start_y=row_y, breakout_length=0.8626, index=i)
                    for i, x in enumerate(xs)]
    return board


def fan_to_top_placement(board: BoardSpec, num_traces: int,
                         rows: int = 2) -> List[Tuple[float, float]]:
    """Planar fan to the top: place test points in `rows` rows across the upper
    region and match pins<->TPs by x (non-crossing). Returns one TP per trace
    (in trace-index order). For load_edge_board."""
    per = -(-num_traces // rows)
    xlo = board.x_min + TP_TO_EDGE_MIN + 5.0
    xhi = board.x_max - TP_TO_EDGE_MIN - 5.0
    top = board.y_max - TP_TO_EDGE_MIN
    start_top = max(t.start_y for t in board.traces)
    rowys = np.linspace(start_top + 0.55 * (top - start_top), top, rows)
    xs = np.linspace(xlo, xhi, per)
    tps = sorted([(float(x), float(y)) for y in rowys for x in xs],
                 key=lambda p: p[0])[:num_traces]
    pins = sorted(range(num_traces), key=lambda i: board.traces[i].start_x)
    placed = [None] * num_traces
    for k, i in enumerate(pins):
        placed[i] = tps[k] if k < len(tps) else tps[-1]
    return placed


def load_edge_board_2row(num_traces: int = 20, board_w: float = 240.0,
                         board_h: float = 250.0, conn_yf: float = 0.38,
                         seed: int = None) -> BoardSpec:
    """Two-row edge connector low on the board: the LOWER row escapes DOWNWARD and
    wraps up the sides, the UPPER row escapes straight up — all traces ending at
    test points in the upper region. Exploratory: the single-layer wrap of the
    lower row is hard, so not all 20 route (see docs/ROUTING.md)."""
    board = BoardSpec(origin_x=0.0, origin_y=0.0, width=board_w, height=board_h)
    cx = board_w / 2.0
    if seed is not None:
        cx += np.random.RandomState(seed).uniform(-board_w * 0.06, board_w * 0.06)
    ccy = board_h * conn_yf
    per = num_traces // 2
    sp = TRACE_MIN_CENTER_TO_CENTER
    xs = [cx + (i - (per - 1) / 2) * sp for i in range(per)]
    span = (per - 1) * sp
    nrz_h = 6.64
    board.rect_obstacles.append(Obstacle(
        cx=cx, cy=ccy, width=span + 4.0, height=nrz_h,
        clearance=TRACE_TO_EDGE_MIN, name="non_routing_zone"))
    board.circ_obstacles.append(CircularObstacle(
        cx=cx, cy=ccy, radius=0.95, clearance=TRACE_TO_UPTH_MIN, name="UPTH_1"))
    board.connector_x = cx - span / 2 - 3.0
    board.connector_y = ccy - (nrz_h / 2 + 3.0)
    board.connector_w = span + 6.0
    board.connector_h = nrz_h + 6.0
    lower_y = ccy - nrz_h / 2 - 0.1          # escapes downward (wraps up the sides)
    upper_y = ccy + nrz_h / 2 + 0.1          # escapes straight up
    traces = [TraceSpec(start_x=x, start_y=lower_y, breakout_length=0.8626, index=i)
              for i, x in enumerate(xs)]
    traces += [TraceSpec(start_x=x, start_y=upper_y, breakout_length=0.8626, index=per + i)
               for i, x in enumerate(xs)]
    board.traces = traces[:num_traces]
    return board


def wrap_to_top_placement(board: BoardSpec, num_traces: int) -> List[Tuple[float, float]]:
    """Planar fan to the top for a 2-row edge board (load_edge_board_2row): the
    lower row goes to the OUTER top test points (wrapping around the sides), the
    upper row to the MIDDLE ones, matched by x so the routing is non-crossing."""
    per = num_traces // 2
    ccy = board.connector_y + board.connector_h / 2
    xlo = board.x_min + TP_TO_EDGE_MIN + 5.0
    xhi = board.x_max - TP_TO_EDGE_MIN - 5.0
    top = board.y_max - TP_TO_EDGE_MIN
    rowys = np.linspace(ccy + 0.55 * (top - ccy), top, 2)
    xs = np.linspace(xlo, xhi, (num_traces + 1) // 2)
    tps = sorted([(float(x), float(y)) for y in rowys for x in xs],
                 key=lambda p: p[0])[:num_traces]
    lower = sorted(range(per), key=lambda i: board.traces[i].start_x)
    upper = sorted(range(per, num_traces), key=lambda i: board.traces[i].start_x)
    h = per // 2
    placed = [None] * num_traces
    for k in range(h):
        placed[lower[k]] = tps[k]                         # lower-left -> leftmost TPs
    for k in range(per):
        placed[upper[k]] = tps[h + k]                     # upper row -> middle TPs
    for k in range(per - h):
        placed[lower[h + k]] = tps[h + per + k]           # lower-right -> rightmost TPs
    return placed


def spread_placement(board: BoardSpec, num_traces: int):
    """Endpoints spread AROUND the connector as a non-crossing radial fan: pick
    well-separated candidate positions and match pins<->TPs by angle. Spread by
    angle (not by maximizing distance), so the test points are distributed but stay
    at similar radius — keeping trace lengths close enough that post-hoc
    equalization works. Endpoints are NOT bucketed into per-layer rows; the router
    (route_auto_layers) assigns layers itself."""
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    ccx = board.connector_x + board.connector_w / 2
    ccy = board.connector_y + board.connector_h / 2
    chosen = []
    # farthest-from-connector first only to seed a spread set; >=13mm apart.
    for idx in np.argsort(-np.hypot(cand[:, 0] - ccx, cand[:, 1] - ccy)):
        if len(chosen) >= num_traces:
            break
        if check_tp_spacing(chosen, *cand[idx]):
            chosen.append(tuple(cand[idx]))
    tps = sorted(chosen[:num_traces], key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(num_traces),
                  key=lambda i: np.arctan2(board.traces[i].start_y - ccy,
                                           board.traces[i].start_x - ccx))
    placed = [None] * num_traces
    for k, i in enumerate(pins):
        placed[i] = tps[k] if k < len(tps) else tps[-1]
    return placed


def equal_length_placement(board: BoardSpec, num_traces: int):
    """Spread test points on a common-radius ring around the connector (angularly
    distributed, matched pin<->TP by angle). Because every TP is ~equidistant from
    the connector, traces come out near-equal length, so they equalize far better
    than a board-filling spread — the real lever for length matching, since post-hoc
    meandering is space-limited (measured: raw spread ~0.31 vs ~0.44, equalized ~0.19
    vs ~0.39). Non-crossing radial fan; the router assigns layers."""
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    ccx = board.connector_x + board.connector_w / 2
    ccy = board.connector_y + board.connector_h / 2
    dist = np.hypot(cand[:, 0] - ccx, cand[:, 1] - ccy)
    R = float(np.median(dist))                          # common target radius
    chosen = []
    for idx in np.argsort(np.abs(dist - R)):            # candidates nearest that radius
        if len(chosen) >= num_traces:
            break
        if check_tp_spacing(chosen, *cand[idx]):
            chosen.append(tuple(cand[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(num_traces),
                  key=lambda i: np.arctan2(board.traces[i].start_y - ccy,
                                           board.traces[i].start_x - ccx))
    placed = [None] * num_traces
    for k, i in enumerate(pins):
        placed[i] = tps[k] if k < len(tps) else tps[-1]
    return placed


@dataclass
class ChallengeSpec:
    """Fully parametric 'moat' challenge board: a ring of obstacles around the connector
    with only `n_gaps` gaps, so traces must funnel through. Every knob is exposed."""
    board_size: float = 120.0
    num_traces: int = 20
    n_gaps: int = 3
    gap_halfwidth: float = 0.45        # gap angular half-width (radians)
    moat_radius_frac: float = 0.30     # moat ring radius / board_size
    obstacle_frac: float = 0.055       # moat obstacle side / board_size
    moat_segments: int = 22            # candidate obstacle slots around the ring
    tp_radius_mult: float = 1.25       # pads placed beyond moat_radius * this
    seed: int = 6
    placement: str = "ring"            # pad POSITIONS: "ring" | "gap_aligned"
    assignment: str = "angle"          # ring pin<->pad matching: "angle" | "gap_aware"


def make_challenge(spec: ChallengeSpec):
    """Build a parametric moat board + a pad placement. Returns (board, placed).

    placement="ring": pads spread on a far ring outside the moat (matched by
      `assignment`: "angle" or gap-grouped "gap_aware").
    placement="gap_aligned": pads CLUSTERED in radial fans at the gaps and pins
      grouped by gap, so each trace shoots straight pin -> nearest gap -> pad with
      minimal crossing (the 'intelligent placement' that makes routing easier)."""
    n, ng = spec.num_traces, spec.n_gaps
    board = load_te_example(num_traces=n, seed=spec.seed, board_size=spec.board_size)
    cx = board.connector_x + board.connector_w / 2
    cy = board.connector_y + board.connector_h / 2
    R = spec.board_size * spec.moat_radius_frac
    gaps = [2 * np.pi * g / ng for g in range(ng)]
    side = spec.board_size * spec.obstacle_frac
    for i in range(spec.moat_segments):
        a = 2 * np.pi * i / spec.moat_segments
        if any(abs(((a - g + np.pi) % (2 * np.pi)) - np.pi) < spec.gap_halfwidth for g in gaps):
            continue                                          # leave the gaps open
        board.rect_obstacles.append(Obstacle(
            cx=cx + R * np.cos(a), cy=cy + R * np.sin(a),
            width=side, height=side, clearance=TRACE_TO_TRACE_MIN, name="moat"))
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    out = cand[np.hypot(cand[:, 0] - cx, cand[:, 1] - cy) > R * spec.tp_radius_mult]
    ang = lambda x, y: np.arctan2(y - cy, x - cx)
    near_gap = lambda x, y: min(range(ng),
        key=lambda g: abs(((ang(x, y) - gaps[g] + np.pi) % (2 * np.pi)) - np.pi))

    if spec.placement == "gap_aligned":
        oa = np.arctan2(out[:, 1] - cy, out[:, 0] - cx)
        per = [n // ng + (1 if g < n % ng else 0) for g in range(ng)]   # pads per gap
        chosen = []
        for g in range(ng):                                   # cluster pads near each gap
            angd = np.abs(((oa - gaps[g] + np.pi) % (2 * np.pi)) - np.pi)
            cnt = 0
            for idx in np.argsort(angd):
                if cnt >= per[g]:
                    break
                if check_tp_spacing(chosen, *out[idx]):
                    chosen.append(tuple(out[idx])); cnt += 1
        for idx in np.argsort(-np.hypot(out[:, 0] - cx, out[:, 1] - cy)):  # fallback fill
            if len(chosen) >= n:
                break
            if check_tp_spacing(chosen, *out[idx]):
                chosen.append(tuple(out[idx]))
        tps = sorted(chosen, key=lambda p: (near_gap(*p), ang(*p)))
        pins = sorted(range(n), key=lambda i: (near_gap(board.traces[i].start_x, board.traces[i].start_y),
                                               ang(board.traces[i].start_x, board.traces[i].start_y)))
    else:                                                      # "ring"
        chosen = []
        for idx in np.argsort(-np.hypot(out[:, 0] - cx, out[:, 1] - cy)):
            if len(chosen) >= n:
                break
            if check_tp_spacing(chosen, *out[idx]):
                chosen.append(tuple(out[idx]))
        if spec.assignment == "gap_aware":
            tps = sorted(chosen, key=lambda p: (near_gap(*p), ang(*p)))
            pins = sorted(range(n), key=lambda i: (near_gap(board.traces[i].start_x, board.traces[i].start_y),
                                                   ang(board.traces[i].start_x, board.traces[i].start_y)))
        else:
            tps = sorted(chosen, key=lambda p: ang(*p))
            pins = sorted(range(n), key=lambda i: ang(board.traces[i].start_x, board.traces[i].start_y))

    placed = [None] * n
    for k, i in enumerate(pins):
        placed[i] = tps[k] if k < len(tps) else tps[-1]
    return board, placed


def challenge_board(board_size: float = 120.0, num_traces: int = 20, n_gaps: int = 3,
                    seed: int = 6, assignment: str = "angle", placement: str = "ring"):
    """Convenience wrapper around make_challenge/ChallengeSpec (back-compat)."""
    return make_challenge(ChallengeSpec(
        board_size=board_size, num_traces=num_traces, n_gaps=n_gaps, seed=seed,
        assignment=assignment, placement=placement))


def generate_candidate_grid(board: BoardSpec, resolution: float = 6.5,
                            max_candidates: int = MAX_CANDIDATES
                            ) -> Tuple[np.ndarray, int]:
    """
    Generate valid test point candidate positions.

    Returns:
        candidates: array of shape (max_candidates, 2) with (x, y) positions,
                    padded with (x_min, y_min) entries beyond the real count.
        real_count: number of genuine (non-padding) candidates.
    """
    candidates = []

    x_lo = board.x_min + TP_TO_EDGE_MIN
    x_hi = board.x_max - TP_TO_EDGE_MIN
    y_lo = board.y_min + TP_TO_EDGE_MIN
    y_hi = board.y_max - TP_TO_EDGE_MIN

    xs = np.arange(x_lo, x_hi + resolution / 2, resolution)
    ys = np.arange(y_lo, y_hi + resolution / 2, resolution)

    for x in xs:
        for y in ys:
            if _is_valid_tp_position(board, x, y):
                candidates.append((x, y))

    # Subsample uniformly if too many (preserve spatial coverage of the whole
    # board instead of head-truncating one side).
    if len(candidates) > max_candidates:
        idx = np.linspace(0, len(candidates) - 1, max_candidates).astype(int)
        candidates = [candidates[i] for i in idx]

    real_count = len(candidates)

    # Pad to fixed size with dummy entries
    while len(candidates) < max_candidates:
        candidates.append((board.x_min, board.y_min))

    return np.array(candidates, dtype=np.float64), real_count


def _is_valid_tp_position(board: BoardSpec, x: float, y: float) -> bool:
    """Check if (x, y) is a valid test point position."""
    # Board edge clearance
    if (x - board.x_min < TP_TO_EDGE_MIN or board.x_max - x < TP_TO_EDGE_MIN or
            y - board.y_min < TP_TO_EDGE_MIN or board.y_max - y < TP_TO_EDGE_MIN):
        return False

    # Connector outline clearance
    conn_xmin = board.connector_x
    conn_xmax = board.connector_x + board.connector_w
    conn_ymin = board.connector_y
    conn_ymax = board.connector_y + board.connector_h
    dx = max(conn_xmin - x, 0, x - conn_xmax)
    dy = max(conn_ymin - y, 0, y - conn_ymax)
    if dx == 0 and dy == 0 and conn_xmin <= x <= conn_xmax and conn_ymin <= y <= conn_ymax:
        return False  # inside connector
    dist_to_conn = np.sqrt(dx**2 + dy**2)
    if dist_to_conn < TP_TO_CONNECTOR_MIN:
        return False  # too close to connector

    # Rectangular obstacles
    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        buf = obs.clearance
        if xmin - buf < x < xmax + buf and ymin - buf < y < ymax + buf:
            return False

    # Circular obstacles
    for obs in board.circ_obstacles:
        dist = np.sqrt((x - obs.cx)**2 + (y - obs.cy)**2)
        if dist < obs.radius + obs.clearance:
            return False

    return True


def check_tp_spacing(placed_tps: List[Tuple[float, float]], x: float, y: float) -> bool:
    """Check if new TP at (x,y) satisfies spacing with all placed TPs."""
    for px, py in placed_tps:
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        if dist < TP_TO_TP_MIN:
            return False
    return True