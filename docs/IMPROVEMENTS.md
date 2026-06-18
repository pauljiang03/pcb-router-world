# Improvement Plan — pcb-router-world

Prioritized, concrete plan to make the project correct, scientifically valid, and
publishable. Companion to `EVALUATION.md` (which explains *why* each item
matters). Effort tags: **S** ≈ <1h, **M** ≈ a few hours, **L** ≈ a day+.

> None of these require a long training run to implement. The only step that
> needs real compute is the final verification, isolated in "When you have a GPU"
> at the bottom.

---

## Status (this session)

**Done & verified, no compute:** P0.1 board randomization · P0.2 corner-TP
guardrail · P0.3 *quick* (reward-only masking + guardrail; docs corrected) ·
P0.4 A* default (**router decision: A\* for training, FreeRouting eval-only**) ·
P1.2 held-out eval seeds · P2 router accuracy (uniform resampling) · P2 tests
(`tests/`, 14 passing) · P2 eval runs without matplotlib (`--no-plot`) ·
P2 docs. Reward components are now logged in `info["reward_components"]`, and the
eval validity metric now requires `failures==0`.

**Routing & 20-trace board overhaul (done & verified — see `ROUTING.md`):** the
board is a central connector with 20 traces (2×10 rows) so traces fan into both
halves; the router is **octilinear (45°) negotiated rip-up-reroute + escape-carving
+ multi-start**, with a **guaranteed-planar (zero-crossing)** result — diagonal
X-crossings are penalized during routing, then any residual is removed using the
verifier's own predicate. A non-crossing angular placement routes ~20/20 at 45°
with 0 crossings. `ROUTING.md` lays out the novel learned-guidance approaches.

**Length matching + reward rebalance (done):** `envs/routing.py:equalize_lengths`
is the crossing-safe Stage-3 meander (spread 0.35→0.02 on a feasible board, 0
crossings). The episode-end reward was rebalanced (no single term dominates) and
now optimizes shorter total length, equal length (spread), and compactness/
containment; `failures` already penalize crossing-requiring placements. All
components logged in `info["reward_components"]`.

**Deferred — need a training run to validate:** P0.3 *faithful* −∞ logit masking
through the actor · P1.3 trained-policy eval path (needs a checkpoint) · P1.4 PPO
baseline (needs stable-baselines3 + compute). (Reward weights are balanced and
sensible, but their *final* tuning still needs a training run.)

---

## P0 — Correctness (do before the next training run)

### P0.1 Make per-episode board randomization actually work  · M
**Problem:** `reset()` never rebuilds the board, so every episode (train *and*
eval) uses one fixed layout (`EVALUATION.md §1`).
**Fix:** move board + candidate-grid construction into `reset()`, seeded per
episode. In `envs/pcb_env.py`:

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:                       # rebuild layout for this episode
        self.board = load_te_example(self._num_traces_requested, seed=seed)
        self.num_traces = min(self._num_traces_requested, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]
        self.candidates, self._real_count = generate_candidate_grid(
            self.board, self._candidate_resolution, MAX_CANDIDATES)
        self._x_scale = (IMG_SIZE - 1) / max(self.board.width, 1e-6)
        self._y_scale = (IMG_SIZE - 1) / max(self.board.height, 1e-6)
    self.placed_tps = []
    self.current_trace = 0
    self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
    self.candidate_mask[self._real_count:] = False
    self.routed_paths = self.routed_lengths = None
    return self._render_obs(), self._get_info()
```

`dreamer_wrapper.py` already passes an incrementing seed — it will now take
effect. **Decision to make first:** if you actually want a *single-board MVP*
(README §13), keep one board and delete the misleading `self._seed += 1` plumbing
instead. Pick one; don't ship both.

### P0.2 Stop placing junk corner TPs for invalid/padding actions  · S
**Problem:** invalid actions still append `(x_min, y_min)` (`EVALUATION.md §3`).
**Fix (guardrail):** in `envs/pcb_env.py:step()`, if the action is masked, snap to
the nearest valid candidate (or no-op the placement) rather than appending a
corner point:

```python
if not self.candidate_mask[action]:
    valid = np.where(self.candidate_mask[:self._real_count])[0]
    if len(valid):
        action = int(valid[np.argmin([
            (self.candidates[v][0]-self.candidates[action][0])**2 +
            (self.candidates[v][1]-self.candidates[action][1])**2 for v in valid])])
    reward -= 2.0
    tp_x, tp_y = self.candidates[action]
```

### P0.3 Decide how masking is enforced  · S (reward-only) / L (true masking)
**Problem:** docs claim −∞ logit masking; the agent has no mask (`EVALUATION.md §2`).
- **Quick (S):** keep masking as reward-only + the P0.2 guardrail, and update the
  docs to say so. Honest and low-risk.
- **Faithful (L):** add `mask` to the obs dict in `dreamer_wrapper.py`, thread it
  through the encoder, and apply `logits = logits.masked_fill(~mask, -1e9)` in the
  actor head (`networks.py`/`models.py`). This is the version that matches the
  paper claim and is worth it if the world-model-vs-baseline result is the thesis.

### P0.4 Default training to A*, reserve FreeRouting for eval  · S
**Problem:** `use_freerouting=True` default makes training try a ~3 s Java
subprocess per episode (`EVALUATION.md §4`).
**Fix:** in `envs/dreamer_wrapper.py`, build
`TPPlacementEnv(num_traces=…, seed=…, use_freerouting=False)`. Use FreeRouting
only in `eval.py --freerouting`.

---

## P1 — Scientific validity

### P1.1 Re-balance the reward  · M
The `-80 * spread` term swamps everything (`EVALUATION.md §5`). Steps:
1. Log each reward component separately for a few greedy/random episodes to see
   the real magnitudes.
2. Normalize terms to comparable scales (e.g. clip `spread` to its achievable
   range and use a coefficient ~1–5), so routability/length/spacing can actually
   influence the gradient.
3. Re-sync the reward tables in `README.md §6` and `plan.md §5` to the code (single
   source of truth — consider generating the doc table from the constants).

### P1.2 Hold out eval boards from training  · S (after P0.1)
Give eval envs disjoint seeds from train (`train.py:101–104`), e.g. train on
seeds `[0, 10_000)` and eval on `[10_000, 10_100)`. This is the only way the
"generalization" claim becomes measurable.

### P1.3 Add a trained-policy path to `eval.py`  · M
`eval.py` only runs random/greedy/spread. Add `--checkpoint` to load `latest.pt`,
roll out the Dreamer policy on the same boards, and print the same metric table —
so "does RL beat baselines?" can be answered with one command and dropped into
`eval_results/`.

### P1.4 Add the PPO sanity baseline  · M
`plan.md §14` promises a PPO fallback. A short `train_ppo.py`
(`stable_baselines3.PPO("CnnPolicy", env)`) on the same env validates that the
environment is learnable at all and is the natural model-free comparison for the
core hypothesis. Cheap to run relative to Dreamer.

---

## P2 — Engineering & polish

- **Tests (M):** convert `test_env.py` to `pytest`; add unit tests for
  `_is_valid_tp_position`, `check_tp_spacing`, candidate-grid validity, and a
  routing-clearance assertion on a known board.
- **Router accuracy (S):** in `validate_routing_constraints` use uniform
  resampling (or full resolution) instead of `[::len//200]` head-stride; make
  `generate_candidate_grid` subsample uniformly instead of head-truncating.
- **Docs/layout (S):** add a short "Quickstart" to `README.md` reflecting the new
  flat layout (`python train.py --configs defaults debug --device cpu`); the old
  Colab steps still say to upload/`cd` into `pcb-dreamer/` — update to
  `pcb-router-world/`. Reconcile the obs/`imag_horizon` drift noted in
  `EVALUATION.md §6`.
- **Artifacts (S):** `obs_initial.png`/`obs_final.png` and `eval_results/*.png`
  are tracked generated outputs; consider moving to a `docs/` folder or
  gitignoring them. `logdir/` is now correctly ignored.
- **Pin deps (S):** pin `torch`, `numpy`, `gymnasium` versions for reproducibility
  (note: code is already numpy-2.x compatible).

---

## Suggested order

1. P0.1 → P0.2 → P0.4 (env correctness; all S/M, no compute).
2. P0.3 quick version + P1.1 reward rebalance (decide thesis scope here).
3. P1.2 / P1.3 / P1.4 (make results measurable).
4. P2 polish.

## When you have a GPU (verification only)

- **First gate (minutes):** `python train.py --configs defaults debug --device cpu`
  — the `debug` config runs 1000 steps; confirm `model_loss` is finite from the
  first log (`plan.md §13`). This is the lightest possible end-to-end check.
- **Real run:** after P0/P1, train with the `defaults` config on GPU and compare
  `eval_return` against the random/greedy/spread baselines (P1.3) and PPO (P1.4).
- The **cheapest real evaluation needs no GPU** and already works today:
  `python eval.py --episodes 5 --num_traces 10` (numpy + A*), which regenerates
  the baseline plots in `eval_results/`.
