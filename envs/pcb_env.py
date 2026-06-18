"""
PCB Test Point Placement Environment.

The agent places test points one at a time, one per trace.
Trace i gets TP i — placement order is the assignment.
After all TPs are placed, a validation loop routes all traces
and computes the reward.

Two routing modes:
  - A* (DEFAULT): fast cell-based router for training (~100ms). No external
    dependencies — chosen as the standard router for both training and the
    no-dependency path.
  - FreeRouting: industry autorouter for optional high-fidelity evaluation
    (~3s, requires Java + freerouting.jar). Opt-in only via use_freerouting=True.

Observation: 64x64x3 uint8 image.
  Red:   obstacles + clearance zones + board edge
  Green: placed test points + exclusion zones + routed traces
  Blue:  current trace starting point + valid remaining candidates
Action: discrete index into candidate grid (fixed size MAX_CANDIDATES).

Note on action masking: the candidate_mask is maintained internally and used
to (a) penalize invalid picks (-2 reward) and (b) snap an invalid pick to the
nearest still-valid candidate so a junk corner TP is never placed. The mask is
NOT yet exposed to the agent as -inf logits (see IMPROVEMENTS.md P0.3).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Tuple

from envs.board import (
    BoardSpec, load_te_example, generate_candidate_grid,
    check_tp_spacing, TP_TO_TP_MIN, TP_TO_EDGE_MIN, MAX_CANDIDATES,
)
from envs.routing import route_all_traces as route_astar

IMG_SIZE = 64


class TPPlacementEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board: Optional[BoardSpec] = None,
        num_traces: int = 10,
        candidate_resolution: float = 6.5,
        use_freerouting: bool = False,
        render_mode: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.use_freerouting = use_freerouting
        self._num_traces_requested = num_traces
        self._candidate_resolution = candidate_resolution
        self._board_seed = seed
        self._board_given = board is not None

        if board is None:
            board = load_te_example(num_traces=num_traces, seed=seed)
        self.board = board
        self.num_traces = min(num_traces, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]

        self.candidates, self._real_count = generate_candidate_grid(
            self.board, candidate_resolution, MAX_CANDIDATES
        )
        self.num_candidates = MAX_CANDIDATES

        self.action_space = spaces.Discrete(self.num_candidates)
        self.observation_space = spaces.Box(
            0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )

        self._update_scales()

        self.placed_tps: List[Tuple[float, float]] = []
        self.current_trace: int = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
        # Mask out padding entries
        self.candidate_mask[self._real_count:] = False

        # Filled after validation
        self.routed_paths = None
        self.routed_lengths = None
        self._reward_components: dict = {}

    # ---- coordinate / drawing helpers ----

    def _update_scales(self):
        self._x_scale = (IMG_SIZE - 1) / max(self.board.width, 1e-6)
        self._y_scale = (IMG_SIZE - 1) / max(self.board.height, 1e-6)

    def _w2p(self, x: float, y: float) -> Tuple[int, int]:
        px = int((x - self.board.x_min) * self._x_scale)
        py = int((y - self.board.y_min) * self._y_scale)
        return np.clip(px, 0, IMG_SIZE - 1), np.clip(py, 0, IMG_SIZE - 1)

    def _draw_circle(self, img, cx, cy, r_mm, ch, val=255):
        pcx, pcy = self._w2p(cx, cy)
        pr = max(1, int(r_mm * self._x_scale))
        for dy in range(-pr, pr + 1):
            for dx in range(-pr, pr + 1):
                if dx * dx + dy * dy <= pr * pr:
                    py, px = pcy + dy, pcx + dx
                    if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                        img[py, px, ch] = min(255, int(img[py, px, ch]) + val)

    def _draw_rect(self, img, xmin, ymin, xmax, ymax, ch, val=255):
        px0, py0 = self._w2p(xmin, ymin)
        px1, py1 = self._w2p(xmax, ymax)
        py0, py1 = max(0, min(py0, py1)), min(IMG_SIZE, max(py0, py1) + 1)
        px0, px1 = max(0, min(px0, px1)), min(IMG_SIZE, max(px0, px1) + 1)
        img[py0:py1, px0:px1, ch] = np.minimum(
            255, img[py0:py1, px0:px1, ch].astype(np.int16) + val
        ).astype(np.uint8)

    # ---- observation ----

    def _render_obs(self) -> np.ndarray:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # RED: obstacles + edge clearance + connector
        for obs in self.board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            self._draw_rect(img, xn - obs.clearance, yn - obs.clearance,
                            xx + obs.clearance, yx + obs.clearance, 0, 150)
            self._draw_rect(img, xn, yn, xx, yx, 0, 255)
        for obs in self.board.circ_obstacles:
            self._draw_circle(img, obs.cx, obs.cy,
                              obs.radius + obs.clearance, 0, 150)
            self._draw_circle(img, obs.cx, obs.cy, obs.radius, 0, 255)
        edge_px = max(1, int(TP_TO_EDGE_MIN * self._x_scale))
        img[:edge_px, :, 0] = 100
        img[-edge_px:, :, 0] = 100
        img[:, :edge_px, 0] = 100
        img[:, -edge_px:, 0] = 100
        if self.board.connector_w > 0:
            self._draw_rect(img, self.board.connector_x, self.board.connector_y,
                            self.board.connector_x + self.board.connector_w,
                            self.board.connector_y + self.board.connector_h,
                            0, 180)

        # GREEN: placed TPs + exclusion zones
        for tx, ty in self.placed_tps:
            self._draw_circle(img, tx, ty, TP_TO_TP_MIN / 2, 1, 60)
            self._draw_circle(img, tx, ty, 1.5, 1, 255)

        # BLUE: current trace start + valid candidates
        if self.current_trace < self.num_traces:
            t = self.board.traces[self.current_trace]
            self._draw_circle(img, t.start_x, t.start_y, 3.0, 2, 255)
        for i in range(self._real_count):  # only draw real candidates
            if self.candidate_mask[i]:
                cx, cy = self.candidates[i]
                px, py = self._w2p(cx, cy)
                if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                    img[py, px, 2] = 150

        # Dim starting points (all traces) in red
        for t in self.board.traces:
            px, py = self._w2p(t.start_x, t.start_y)
            if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                img[py, px, 0] = min(255, int(img[py, px, 0]) + 80)

        return img

    # ---- candidate mask ----

    def _update_candidate_mask(self):
        for i in range(self._real_count):  # only check real candidates
            if self.candidate_mask[i]:
                cx, cy = self.candidates[i]
                if not check_tp_spacing(self.placed_tps, cx, cy):
                    self.candidate_mask[i] = False

    def _nearest_valid_candidate(self, action: int) -> int:
        """Map an invalid/padding action to the nearest still-valid real
        candidate so a junk corner TP is never placed. Falls back to the
        nearest real candidate if none remain valid (never returns padding)."""
        ref = self.candidates[action]
        valid = np.where(self.candidate_mask[:self._real_count])[0]
        pool = valid if len(valid) else np.arange(self._real_count)
        if len(pool) == 0:
            return action
        d = ((self.candidates[pool, 0] - ref[0]) ** 2 +
             (self.candidates[pool, 1] - ref[1]) ** 2)
        return int(pool[int(np.argmin(d))])

    # ---- validation (runs after all TPs placed) ----

    def _validate(self) -> float:
        """Route all traces and compute reward."""
        if self.use_freerouting:
            try:
                from envs.freerouting import route_with_freerouting
                paths, lengths, failures = route_with_freerouting(
                    self.board, self.placed_tps
                )
            except FileNotFoundError:
                import warnings
                warnings.warn(
                    "FreeRouting not found, falling back to A*. "
                    "Set FREEROUTING_JAR env var or place freerouting.jar "
                    "in project root.",
                    stacklevel=2,
                )
                self.use_freerouting = False
                paths, lengths, failures = route_astar(
                    self.board, self.placed_tps
                )
        else:
            paths, lengths, failures = route_astar(
                self.board, self.placed_tps
            )

        self.routed_paths = paths
        self.routed_lengths = lengths

        reward = 0.0
        comp = {}
        diag = np.hypot(self.board.width, self.board.height)

        # Feasibility: full routing means every trace connected, no crossing-forced
        # drops (the router drops a net rather than cross — so this also penalizes
        # placements that require crossings).
        comp["routable"] = 10.0 if failures == 0 else -5.0 * failures
        reward += comp["routable"]

        finite = [l for l in lengths if l < float('inf')]
        if finite:
            # (1) Minimize total trace length (normalized per routed trace).
            comp["length"] = -10.0 * sum(finite) / (len(finite) * diag)
            reward += comp["length"]
            # (2) Equal length: penalize length spread so post-hoc meandering
            #     (envs.routing.equalize_lengths) can equalize within available space.
            if len(finite) > 1:
                spread = (max(finite) - min(finite)) / max(np.mean(finite), 1e-6)
                comp["spread"] = -8.0 * spread
                reward += comp["spread"]

        # (3) Compactness / containment: reward a small test-point footprint.
        if len(self.placed_tps) > 1:
            xs = [p[0] for p in self.placed_tps]
            ys = [p[1] for p in self.placed_tps]
            bbox_diag = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
            comp["compactness"] = -5.0 * bbox_diag / diag
            reward += comp["compactness"]

        # Per-component breakdown for reward-balance diagnostics (logged in info).
        # Weights kept comparable so no single term dominates (the old -80*spread did).
        self._reward_components = comp
        return reward

    # ---- gym interface ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Regenerate the board layout for this episode so different seeds give
        # different layouts (board randomization). Skipped when an explicit
        # board was passed in or no seed is given.
        if seed is not None and not self._board_given:
            self.board = load_te_example(
                num_traces=self._num_traces_requested, seed=seed
            )
            self.num_traces = min(self._num_traces_requested, len(self.board.traces))
            self.board.traces = self.board.traces[:self.num_traces]
            self.candidates, self._real_count = generate_candidate_grid(
                self.board, self._candidate_resolution, MAX_CANDIDATES
            )
            self._update_scales()

        self.placed_tps = []
        self.current_trace = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
        self.candidate_mask[self._real_count:] = False  # mask padding
        self.routed_paths = None
        self.routed_lengths = None
        self._reward_components = {}
        return self._render_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        reward = 0.0

        # Per-step reward. Invalid (masked or padding) picks are penalized and
        # snapped to the nearest still-valid candidate so we never place a junk
        # corner TP.
        if not self.candidate_mask[action]:
            reward -= 2.0
            action = self._nearest_valid_candidate(action)
        elif check_tp_spacing(self.placed_tps, *self.candidates[action]):
            reward += 1.0
        else:
            reward -= 2.0

        tp_x, tp_y = self.candidates[action]
        self.placed_tps.append((tp_x, tp_y))
        self.current_trace += 1
        self._update_candidate_mask()

        # Preserve future options (count only real candidates)
        valid_frac = self.candidate_mask[:self._real_count].sum() / max(self._real_count, 1)
        reward += 0.3 * valid_frac

        terminated = self.current_trace >= self.num_traces

        if terminated:
            reward += self._validate()

        return self._render_obs(), reward, terminated, False, self._get_info()

    def _get_info(self):
        info = {
            "current_trace": self.current_trace,
            "traces_placed": len(self.placed_tps),
        }
        if self.routed_lengths is not None:
            info["trace_lengths"] = self.routed_lengths
            info["failures"] = sum(
                1 for l in self.routed_lengths if l == float('inf')
            )
        if self._reward_components:
            info["reward_components"] = dict(self._reward_components)
        return info

    def render(self):
        return self._render_obs()
