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