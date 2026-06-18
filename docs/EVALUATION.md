# Evaluation — pcb-router-world (DreamerV3 Test-Point Placement)

*Static code + design review. No training was run (per request). Findings are
grounded in reading every source file plus a numpy-only smoke test of the
board/router and a determinism check of the env reset.*

## Verdict

A genuinely interesting research framing (generative co-design of test-point
*positions*, not routing order) with a clean, well-documented environment and a
correctly-forked DreamerV3 backbone. The engineering is above-average for a
research prototype. **However, several correctness gaps between the documented
method and the implementation would materially undermine a training run today** —
most importantly, per-episode board randomization is silently dead, the
"invalid-action masking" described throughout the docs is not wired into the
agent, and the reward is numerically dominated by a single term. None are hard to
fix; see `IMPROVEMENTS.md` for a prioritized plan.

## Status — fixes applied this session

The no-compute correctness items are now **implemented and verified** (11 unit
tests pass; the A* evaluation runs end-to-end with no matplotlib/GPU). Resolved:
**#1** board randomization (regenerates per episode — proven by a unit test),
**#3** corner-TP injection (invalid actions snap to the nearest valid candidate),
**#4** router default (**decision: A\*** for training; FreeRouting opt-in for eval
only), **#6** the obs comparison cell + reward-table sync note, plus the
overstated validity metric (now requires `failures==0`) and per-component reward
logging (`info["reward_components"]`). Still open because only a training run can
judge them: **#2** faithful −∞ logit masking, **#5** reward re-balancing
(instrumented, not retuned). Per-finding status is in `IMPROVEMENTS.md`.

Measured A* baselines (TE board, 10 traces, 5 eps): Greedy 303 mm / 2 fails (best),
Random ~572 mm / 3.4 fails, Spread 744 mm / 3 fails — none fully route all 10
traces, which motivates the learned approach.

## What it is

DreamerV3 (model-based RL, forked from `NM512/dreamerv3-torch`) learns to place
N test points on a PCB, one per trace. Observation is a 64×64×3 image; action is
a discrete index into a pre-computed candidate grid. After all placements, traces
are routed (A* or FreeRouting) and an episode-end reward scores routability,
total length, length-spread, and spacing. Three deterministic baselines (random /
greedy / spread) exist for comparison.

## Repo structure (after cleanup)

The double-nesting was fixed: the project was `…/pcb-router-world/pcb-dreamer/<code>`
(a git repo whose entire contents were a single sub-folder). The inner
`pcb-dreamer/` folder was collapsed, so the code now lives directly at the repo
root `pcb-router-world/` (where `.git` and `.gitignore` already were). The stale
`.gitignore` entry `pcb-dreamer/logdir/` → `logdir/` was corrected. This change
is in the working tree and **uncommitted** — review with `git status` and commit
when ready (git will detect the moves as renames).

## What works well

- **Forked DreamerV3 edits are all present and correct.** All 6 modifications
  documented in `plan.md §8` are in place (verified), including the numpy-2.x
  `np.concatenate` fix — which matters, since numpy **2.4.6** is the installed
  version here.
- **All Python files compile** (`py_compile` clean) and the custom `wrappers.py`
  cleanly drops the legacy `gym` dependency (pure `gymnasium`), so
  `requirements.txt` is self-consistent.
- **The A* router is thoughtful**: cell size guarantees trace clearance,
  congestion penalty spreads routes apart, diagonal crossings are forbidden, the
  first step is forced cardinal, and there's rip-up-and-retry over multiple net
  orderings. The numpy-only smoke test routed the greedy placement end-to-end
  (e.g. seed 0: 155 candidates, 2 routing failures, ~287 mm total).
- **Clean data model** (`BoardSpec`/`Obstacle` dataclasses), sensible
  constraint-driven candidate generation, good docstrings, and a working
  `gymnasium` env that passes `check_env`.
- **CPU fallback** in `train.py` (auto-switches off `cuda:0` when unavailable),
  and an optional FreeRouting path for higher-fidelity evaluation.

## Correctness findings (obvious errors first)

### 1. Per-episode board randomization is dead — the agent trains on ONE board  ⚠️ high impact
`TPPlacementEnv.reset()` (`envs/pcb_env.py:230`) resets placements/mask but never
rebuilds `self.board` or `self.candidates`; the board is fixed in `__init__`.
`PCBDreamerEnv.reset()` (`envs/dreamer_wrapper.py:34`) increments `self._seed`
each episode and passes it to `self._inner.reset(seed=…)`, but that seed only
seeds the gym RNG — it does **not** regenerate the layout.
**Evidence (measured):** across 3 resets the seed counter went 0→3 while the NRZ
center and the observation image were *byte-identical*. Consequence: the README's
board randomization (§5) and generalization story (§9) are never exercised.
`train.py` compounds this by giving train and eval envs the **same** `config.seed`
(`train.py:101–104`), so even conceptually there is no held-out board.

### 2. "Invalid-action masking" is documented but not implemented  ⚠️ high impact
README §4–5 and `method.md §4` describe per-step masking with invalid logits set
to −∞. The Dreamer actor (`networks.py` MLP → `tools.OneHotDist`) never receives
a mask, and the obs dict emitted by `dreamer_wrapper.py:37,43` contains only
`image/is_first/is_last/is_terminal`. The env's `candidate_mask`
(`pcb_env.py:75`) is internal-only. The agent therefore samples freely over all
200 actions; invalidity is discouraged *only* by a −2 reward.

### 3. Padding / invalid actions still inject a junk corner test point
`pcb_env.py:241` reads `tp_x, tp_y = self.candidates[action]` and `:252`
appends it **unconditionally**. Padding slots (index ≥ real candidate count) map
to `(x_min, y_min)` — the board corner, which itself violates edge clearance — so
an invalid action both costs −2 *and* places a garbage TP (and repeated padding
picks stack identical corner TPs). This pollutes routing and the episode-end
reward. Combined with #2, nothing prevents it.

### 4. Training defaults to FreeRouting instead of A*
`pcb_env.py:42` defaults `use_freerouting=True`, and `dreamer_wrapper.py:12`
constructs the env without overriding it — yet the module docstring says "A*
(default) for training." If `freerouting.jar`+Java are present, *every* episode
end spawns a ~3 s Java subprocess (fatal for 500k-step training). If absent, each
env wastes one attempt + a warning before falling back to A* (`pcb_env.py:178`).

### 5. Reward is dominated by the length-spread term
`pcb_env.py:215`: `reward -= 80.0 * spread`. Measured greedy `spread ≈ 1.0–1.5`
for this geometry → **−80 to −120**, dwarfing routability (+15 / −10·fails),
total-length (≈ −1), and spacing (+2). This is exactly the "~−140 baseline" noted
in `plan.md §13`. Because the connector start points are tightly clustered while
TPs must fan out, large spread is partly geometry-forced — so the dominant
gradient may push toward degenerate low-spread layouts rather than the stated
length/spacing objectives. The code's weights also disagree with the docs
(code +15/−5/−80 vs README/plan +10/−0.5/−3) — they have drifted.

### 6. Documentation ↔ code drift (won't crash, but misleads)
- README §11 comparison row claims a "Structured vector ~500 floats (MLP)" obs,
  while the build is image+CNN (consistent everywhere else).
- `imag_horizon` is 25 in `configs.yaml` vs 15 in `plan.md`; `time_limit` default
  25 vs episodes = `num_traces`.
- Reward tables in README/plan no longer match `_validate()`.

### 7. Minor / lower-impact
- `validate_routing_constraints` subsamples each path to ≤200 points
  (`routing.py:426`) before the pairwise trace-to-trace check, which can
  under-report the true closest approach (affects the reported metric only).
- `generate_candidate_grid` truncates to the first 200 candidates in
  column-major order (`board.py:287`); for finer grids/larger boards this would
  silently drop one side of the board. No-op for the current TE board.

## Reproducibility / scientific status

- **No trained checkpoint or Dreamer-policy results are in the repo** —
  `eval_results/` holds only random/greedy baseline images. The headline question
  ("does the world model beat baselines?") is not yet demonstrated.
- No automated tests beyond the `test_env.py` smoke script (not pytest).
- `requirements.txt` is unpinned (fine for a prototype).

## Checks performed (cheap, no training)

- `py_compile` on all `.py` — clean.
- numpy-only smoke: candidate grid (155–160 cands), greedy placement + A* routing
  end-to-end with metrics, for seeds {None, 0, 1}.
- Reset-determinism probe proving finding #1.
- Subagent verification of the 6 forked-file edits + masking wiring.
