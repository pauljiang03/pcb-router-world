# Routing for Central-Connector Test-Point Boards
### Techniques, evidence, and intelligent (learned) approaches

*Every number in this document was measured on the actual code (CPU only, no
training). Reproduction commands are in §8.*

---

## TL;DR

- A feasible routing **almost always exists** on a central-connector board (a
  radial fan is the trivial witness). The historical failures were the **router
  not finding it**, not infeasibility — exactly the hypothesis that motivated
  this work.
- **Traces must never cross (single layer).** The router is **octilinear (45°)**
  and enforces planarity two ways: diagonal X-crossings are penalized during
  negotiation, and a final pass drops any net still geometrically crossing
  another (using the *same* predicate as the verifier). So `route_all_traces`
  output has **zero crossings, always** (regression-tested). An earlier version
  reported "0 failures" while silently producing **8 crossing pairs**; fixed.
- With a **non-crossing angular placement** + negotiated rip-up-reroute +
  escape-carving + multi-start, a 20-trace board routes with **0 crossings on
  every seed**, ~0–2 nets dropped, at ~1.5–1.9 m total (45° is ~25 % shorter than
  pure-rectilinear). Slower on hard seeds; tunable via `n_starts`.
- The single most important empirical finding for *future* work: **routing
  guidance (net order / waypoints) is a large, learnable signal.** The *same*
  placement routes with anywhere from **1 to 6 failures purely depending on net
  order**. Search exploits this expensively; a learned policy could exploit it
  directly. That is the novel direction in §4.

---

## 1. The problem and its root cause (with evidence)

The agent places 20 test points; the router must connect each start (clustered
at a central connector) to its TP on a single layer under clearance constraints.

**Claim: solutions exist; the old router just missed them.** Evidence gathered
by instrumenting the routers:

| Probe | Result | Conclusion |
|---|---|---|
| Each net routed **alone** (old sequential A*) | **0/12 fail** | every net is individually routable |
| All nets together (old sequential A*, short placement) | **9/20 fail** | one-shot sequential routing boxes nets out |
| Same, negotiated rip-up-reroute | **0/20 fail** (10 iters) | the *solution was there*; the router has to look harder |
| Same placement, **40 random net orders** | failures range **1–6** (mean 3.9) | outcome is dominated by *order*, i.e. search/guidance |

The mechanism: one-shot sequential A* routes a net, **hard-blocks its whole
corridor**, and moves on. Early nets consume corridors that later nets needed,
so later nets fail even though a globally-feasible assignment exists.

A second, separate bug: the connector keep-out is rasterized as a conservative
bounding box (it blocks **268 more cells** than a tight rasterization), which
**boxed in the start pads** — every A* from those starts returned "no path."

---

## 2. The router we ship now

Four mechanisms, each measured to matter:

### 2.1 Escape carving (fixes boxed-in starts)
Carve a short free stub outward from each start (lower row downward, upper row
upward) so a trace can always leave the connector keep-out.
*Effect:* 8-trace seed 0 went **8→0 failures**; 20-trace seed 0 went **11→1**.

### 2.2 Negotiated congestion (PathFinder-style)
Instead of hard-blocking, nets may share cells provisionally at a `present`
penalty. After each pass, a per-cell `history` cost accumulates on over-used
cells, so contention migrates apart over a few passes. Because each A* pass also
minimizes path length, converged routes are short.
*Effect:* short placement **9→0 failures** vs one-shot A*.

### 2.3 Multi-start (exploits order-sensitivity)
Routing order matters enormously (§1). We try informed orders (identity,
longest-first, shortest-first) then deterministic random restarts, keep the best
(fewest failures, then shortest), and early-exit on the first conflict-free
result.
*Effect (greedy 20-trace, best-of-K):* single→24 starts drove seeds 0,1,3,5 from
{1,3,2,2} failures **to 0**.

### 2.4 Planarity guarantee (45° routing without crossings)
The router moves **octilinearly** (`_NEG_NBR` is 8-connected, for short 45°
routes). Diagonals can cross without sharing a cell, so planarity is enforced two
ways: (1) during negotiation a diagonal move is penalized if the *complementary*
diagonal of its unit square is in use, with history accumulating on squares whose
two diagonals are both used; (2) a final pass (`_remove_crossings`) drops any net
still geometrically crossing another, using the **same predicate as the verifier**
`count_crossings`. So **every returned routing has zero crossings, regardless of
placement** (`test_router_output_never_crosses`).

### 2.5 Benchmark (20 traces, 9 seeds, octilinear, multi-start)

| Placement | Dropped nets | Crossings | Length |
|---|---|---|---|
| **Angular** (non-crossing by construction) | **~0–2** | **0 / 9 seeds** | ~1.5–1.9 m |
| Greedy (clustered) | high | 0 (crossing nets dropped) | — |

**Reading:** with a placement that admits a planar routing (the angular
assignment, §4-adjacent), the router connects ~18–20/20 at 45° with **zero
crossings**. A placement that *requires* crossings (clustered/greedy) is impossible
on one layer, so it surfaces honestly as dropped nets — never as crossings.
Choosing a placement that is non-crossing, short, and compact is the **agent's**
job; the router guarantees a legal planar result or drops the offending nets. (45°
is ~25 % shorter than pure-rectilinear; the remaining length is recovered by
placement quality, not by letting traces cross.)

### 2.6 Endpoint (pad) clearance & no self-encirclement
Each test point is a pad, not just a trace end, so the router reserves a keep-out
disk (radius `TP_CLEARANCE_CELLS`, currently **3 cells ≈ 4 mm**) around every pad
that **only that pad's own net may enter** — keeping every pad clear of all *other*
traces' bodies and of other pads (measured min pad-to-other-trace ≈ **4.2 mm** vs
the ~1.3 mm trace pitch; tunable — 2–4 cells all route the planar board).
`validate_routing_constraints` reports `tp_to_trace_min`. The Stage-3 length-match
meander (`equalize_lengths`) also refuses bumps within a few cells of either
endpoint, so **a trace never surrounds its own pad** with its own body.

### 2.7 Routing to one side (edge connector → opposite edge)
Not a general constraint, but a useful capability. Two layouts:
- **Single row** (`load_edge_board` + `fan_to_top_placement`): a row of pins low on
  the board, all escaping **upward** to test points across the upper region matched
  pin↔TP by x → the planar router routes **all 20 to the top, 0 crossings**.
- **Two rows** (`load_edge_board_2row` + `wrap_to_top_placement`): the lower row
  escapes **downward and wraps up the sides**, the upper row escapes straight up —
  both ending at the top. On a single layer this wrap is hard: the upper row routes
  10/10 but only ~4 of the lower row's 10 fit the wrap, so **~14/20 route, 0
  crossings** (more search / more space / a narrower keep-out don't lift it — it's a
  single-layer limit; a 2-layer board is the real fix). Every routed trace is then
  **length-matched** (`equalize_lengths`), visible as serpentine meanders on the
  short traces (spread **0.9 → ~0.2**, all ≈ 193 mm).

Reproduce: `python scripts/route_to_top.py --rows {1,2}` (figure
`eval_results/router_top_fan.png`). A *central* connector cannot route to one side
at all (its downward-escaping row gets stuck).

### 2.8 Multi-layer routing (automatic layer assignment)
`route_auto_layers` (`python scripts/route_multilayer.py`, figure
`router_multilayer.png`). The non-routing zone is a **keep-out, so it blocks every
layer** — *not* a top-layer-only component. The router assigns layers by
*routability*: route as many nets as fit planar on layer 0, push the rest to the
next layer, repeat. Same-layer crossings stay forbidden; two nets on **different**
layers may cross — an inter-layer **via** at their endpoints. With test points
**spread** around the connector (`spread_placement`, not bucketed into per-layer
rows) a good placement routes on a **single layer, 0 crossings**; denser placements
cascade to more layers and the **via count** is the number of nets on inner layers.
The two levers — **layer assignment** (found here by routability) and **routing
order** within each layer (the multi-start) — are exactly the guidance a learned
policy could supply (§4). (Note: routing all 20 of a 2-row connector to one edge
stays hard even multi-layer, because the non-routing zone *between* the rows blocks
the lower row's wrap on every layer.)

*Tradeoff:* multi-start costs ~`n_starts`× A* runs. It's tunable
(`route_all_traces(..., n_starts=)`); training can use a small value and
evaluation a larger one.

### 2.9 Further improvements — what the harder-board tests showed
Stress-tested on shrinking boards (160 → 95 mm, greedy + spread placements). Results:

**Shipped (measured wins):**
- **Auto multi-layer assignment** (§2.8) — routes **20/20 even at 95 mm**, degrading
  gracefully by adding layers (4 → 5), never failing. Robust on every size tested.
- **Deeper meander teeth** (`equalize_lengths`, `_MAX_BUMP_DEPTH`) — the old meander
  bumped only the *adjacent* cell and plateaued (0.38 at any pass count); teeth that
  reach into 2-D free space help on the tightest boards (95 mm: 0.72 → **0.51**), and
  stay crossing-safe.
- **Equal-length placement** (`equal_length_placement`) — the **real length lever**:
  TPs on a common-radius ring → near-equal traces → spread 0.31 → **0.19**, vs the
  board-filling spread's 0.44 → 0.39 (~2× better). Post-hoc meandering is space-
  limited because the *short* traces sit in the congested region near the connector,
  so length must be controlled at placement time.
- **Exact clearance verifier** (`min_trace_separation`; `trace_to_trace_center_min` +
  `clearance_ok` in `validate_routing_constraints`). The old check used resampled
  points + a loose threshold and **missed sub-pitch spacing**; this is exact.
- **Rectilinear base for guaranteed clearance** (`route_all_traces(diagonal=False)`).
- *(Optional, off by default — angles stay ≤ 45°:* `any_angle_shortcut` can string-pull
  to arbitrary angles while verifying ≥ pitch, but the project keeps routes octilinear.)*

**Finding + fix — 45° at minimum pin pitch can't keep full clearance.** Two 45°
traces leaving pins one pitch apart are inherently `pitch·sin45 ≈ 0.94 mm` apart (< the
1.33 mm pitch) — geometric, not a bug. A *diagonal-safe* grid (pitch·√2) is infeasible
(min-pitch pins then collide); a parallel-diagonal penalty is a no-op (forced). **The
fix that works: a rectilinear base** (`diagonal=False` — axis-adjacent cells are
*exactly* the pitch, so there is no diagonal gap). On a deliberately hard **"moat"
board** (`challenge_board` — obstacles ring the connector with only a few gaps so traces
must funnel through, `python scripts/gen_figures.py`): per-layer min trace separation
goes **0.94 mm (45°, `router_challenge_45deg.png`) → 1.33 mm = pitch (rectilinear,
`router_challenge_rectilinear.png`)**, both **20/20 routed, 2 layers, 0 same-layer
crossings**, rectilinear ~5–15% longer. So clearance is *guaranteed* (≥ pitch) on boards
where traces can't fan straight out, while staying ≤ 45°.

**Tested and rejected:**
- **Min-via layer assignment by crossing-graph coloring** — *worse* than the greedy
  cascade (16 vias vs 12 on the default board); the cascade already packs layer 0
  near-maximally. Not shipped.
- **Parallel-adjacent-diagonal penalty** in the negotiation — a no-op (the parallel
  diagonals in a min-pitch fan are geometrically forced; nowhere to move them).
- **Diagonal-safe (coarser) grid** — infeasible: coarsening to make diagonals safe
  makes min-pitch connector pins share grid cells.

**Still open (untested ideas, roughly by ROI):**
- **Adaptive negotiation** (raise penalties on stagnation);
- **Smarter rip-up** (rip the most-congested crossing net vs dropping it);
- **Differential pairs / matched buses**; **manufacturability** (teardrops, acute-angle
  removal, true geometric DRC);
- **Net ordering** (the lever you flagged) + learned **layer assignment** — order
  shifts the routed count by several nets; `scripts/gen_ordering_data.py` distills a
  learned order (§4.1).

---

## 3. The key insight: routing guidance is a large, learnable signal

The order-variance result is worth restating because it is the whole basis for
the learned approaches:

```
Same board, same placement, 40 random routing orders:
fails:  1 2 3 4 5 6
count:  3 4 6 13 9 5      (min 1, max 6, mean 3.9)
```

A factor-of-6 swing from a single discrete choice (order) means a policy that
picks *good* guidance — order, or where to route through — captures most of the
benefit that brute-force multi-start gets, but in **one shot** instead of K.
This is the opening for a learned model, and it is what your "build with
intermediate points to guide the solution" intuition is really pointing at.

---

## 4. Intelligent (learned) approaches — the novel direction

The unifying idea: **learn to *guide* a cheap deterministic router, rather than
learn to route pixel-by-pixel.** The router (negotiated A*) guarantees clearance
and shortest-within-guidance; the model only supplies the high-leverage discrete
choices. This keeps the hard geometry exact and the learning problem small.

### 4.1 Learned net-ordering by search-distillation  *(highest ROI, start here)*
Train a policy `π(order | board, placements)` to output a routing order that
minimizes failures+length. **Labels are free:** multi-start already finds the
best order per board, so this is *imitation of search* (policy distillation, the
AlphaZero recipe — search is the teacher, the net learns to skip the search).
- *Model:* small pointer/attention network over the N nets (start, TP, geometry
  features). *Reward/loss:* match the best-of-K order, or RL on −(failures, length).
- *Why it works here:* §3 shows order alone swings results 1→6; the signal is
  strong and cheap to label.

### 4.2 Learned intermediate-point (waypoint) guidance  *(your idea, formalized)*
A recurrent policy emits, for each net, a short sequence of **subgoal waypoints**
that the router threads through (start → w₁ → w₂ → … → TP). "Keep adding
intermediate points until you reach the target" becomes a learned **subgoal
generator**: at each step it proposes the next waypoint conditioned on the board,
the nets already routed, and the remaining distance to target — stopping when the
target is directly reachable.
- *Why waypoints and not raw paths:* a waypoint is a tiny action (one cell) yet
  reshapes the whole route; the router fills in clearance-correct geometry
  between waypoints. The model never has to learn clearance — only *where to aim*.
- *Model:* RL policy (or a Dreamer actor) over "place next waypoint" actions;
  reward = did the net reach its target conflict-free, and how short.
- *Relation to §4.1:* ordering is the 0-waypoint special case; this generalizes
  it to spatial guidance for the genuinely congested boards.

### 4.3 Learned congestion cost-field  *(amortize the negotiation)*
A CNN predicts a per-cell **cost map** (an image, like the board observation)
that tells the *first* A* pass where to avoid. The negotiated history cost is
exactly such a field discovered by iteration; a network can predict it directly
from the board+placement image, so one guided pass ≈ many negotiation passes.
- *Labels are free:* the converged `history` field from the current router is the
  regression target. Pure supervised learning, no RL needed to start.

### 4.4 World-model co-design of placement **and** routing guidance  *(ties to this repo's DreamerV3 thesis)*
The project's core claim is *generative co-design*: the agent chooses where TPs
go. Routing guidance folds in naturally — the **world model imagines how a
placement + a guidance choice cascades into routability, length and spacing**,
and the actor co-optimizes both. The router (negotiated A*) is the differentiable-
enough, fast environment that scores each imagined choice. This is the novel
end state: not "RL places, fixed router routes," but **one world model reasoning
jointly over placement and the routing subgoals that make it manufacturable.**

### Comparison

| Approach | Learns | Labels | Cost to build | Expected gain |
|---|---|---|---|---|
| 4.1 Net ordering | a permutation | free (multi-start) | low | recovers most of multi-start in 1 shot |
| 4.2 Waypoint guidance | spatial subgoals | RL / search | medium | handles the hard, congested boards |
| 4.3 Cost-field | a cost image | free (history field) | low–med | fewer negotiation passes (speed + quality) |
| 4.4 World-model co-design | placement + guidance | RL (Dreamer) | high | the project's headline result |

---

## 5. How to train without hand-labels
- **Search-as-teacher (4.1, 4.3):** the multi-start router already produces the
  best order and the converged cost-field per board — use them as supervised
  targets. No human labels, no reward shaping.
- **RL (4.2, 4.4):** reward = `+1` per net routed conflict-free, `−λ·length`,
  small step penalty per waypoint (encourages few intermediate points — directly
  your "as few intermediate points as needed" intuition). The existing
  `info["reward_components"]` plumbing already exposes these signals.

---

## 6. Validation summary (all measured, §8 to reproduce)

| Finding | Evidence |
|---|---|
| Solutions exist; old router missed them | isolation 0/12 fail, but together 9/20 fail |
| Order carries the signal | same placement: **1–6** failures over 40 orders |
| Negotiation recovers solutions | short placement 9→**0** failures |
| Escape carving fixes boxed starts | 8-trace seed 0: 8→**0** |
| Multi-start helps | greedy seeds {0,1,3,5}: {1,3,2,2}→**0** |
| Shipped router on good placement | fan 20-trace: **0.11** mean fails, **1.06×** optimal length |

---

## 7. Roadmap & recommendation
1. **Done:** central 20-trace board; negotiated + carve + multi-start router
   (`envs/routing.py`); ~20/20 on reasonable placements.
2. **Next (cheap, high ROI):** distill multi-start's best orders into a learned
   net-ordering policy (§4.1) — pure imitation, free labels.
3. **Then:** learned cost-field (§4.3, free labels) and waypoint guidance (§4.2).
4. **Headline:** fold routing guidance into the DreamerV3 world model for true
   placement+routing co-design (§4.4).

## 8. Reproduce
```bash
python -m pytest tests/ -q                    # router + env regression (14 tests)
python eval.py --num_traces 20 --no-plot      # baselines + Planar on the 20-trace board
python scripts/gen_ordering_data.py --boards 8 --orders 12   # §4.1 distillation dataset
```
The §1–§3 / §6 experiments (isolation, order-variance, multi-start, fan-vs-greedy
benchmark) were run as standalone CPU scripts against `envs/routing.py` and
`envs/board.py`; each is a few lines using `route_all_traces` and the internal
`_negotiate`/`_astar_cost` helpers. `scripts/gen_ordering_data.py` is the §4.1
data generator: it records the best routing order per board (free labels from
multi-start) for a learned ordering policy.
