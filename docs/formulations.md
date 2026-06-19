# Training formulations (and which to elect)

The router is classical — negotiated octilinear/rectilinear A\* + automatic layer
assignment (`envs/routing.py`). A learned policy can be injected at several points in
the pipeline. This doc defines each as an MDP, says what it targets, and **elects one**.
The encodings live in `envs/formulations.py` (specs + a shared reward + a CPU evaluator
for the elected one). Boards are the **parametric `ChallengeSpec`** (`envs/board.py`).
Heavy RL is GPU/later — this is the spec so you can pick a formulation and wire it up.

## The decision pipeline
```
place test points  ->  match pin<->pad  ->  order nets  ->  assign layers  ->  route each net (A*)
      (F1)                  (F1)            (F2)            (F3)            classical
```
Each arrow is a place a policy could replace; F4 fuses order+layer, F6 fuses all.

## Formulations (`envs/formulations.py:FORMULATIONS`)
| key | policy controls | reward | helps the bottleneck? | cost |
|---|---|---|---|---|
| **placement** (F1) ⭐ | where each pad goes | routable+length+spread+compact | **yes, on boards with room+gaps (measured)** | **already built** (`TPPlacementEnv`) |
| **net_ordering** (F2) | order nets are routed | −failures −crossings −length | directly | dataset exists (`gen_ordering_data.py`) |
| **layer_assignment** (F3) | layer per net | −layers −vias −failures | directly | moderate |
| **ordered_layer** (F4) ◆ | next (net, layer) jointly | −layers −vias −failures −length | **head-on (the hard residual)** | moderate; subsumes F2+F3 |
| **waypoint** (F5) | A\* intermediate points | routed + length | per-net only | moderate |
| **joint_worldmodel** (F6) | placement+order+layer | global quality | everything | largest |

## Election: **F1 (placement) first ⭐, then F4 (ordered layer-assignment) ◆**
Measured (`ROUTING.md §2.9`): **intelligent placement is a strong lever** — clustering
pads in radial fans at the gaps takes 120 mm / 3-gap boards to **single layer at 16 and
18 traces (0 vias)** and 19/20 at 20 traces, vs 15–17 for the spread placement. Since the
repo's RL **already controls placement** (`TPPlacementEnv`) and there's demonstrated
single-layer headroom for a policy to learn, **F1 is the best first thing to train**.

It is **not** universal: on tight (100 mm) or few-gap (2-gap) boards no placement beats
the topology (gap-aligned ties or loses there). For that residual, the deeper lever is
**F4 — ordered layer-assignment**: control routing order + layer jointly so the fewest
nets spill to inner layers. Its reward (`routing_reward`) is exactly `layers + vias +
failures (+ length)`; it subsumes F2 and F3 and extends the net-ordering distillation
(`scripts/gen_ordering_data.py`). Build F4 when placement alone can't make a board
single-layer.

## Shared reward — `envs/formulations.py:routing_reward`
`w_route·routed − w_layer·(layers−1) − w_via·vias − w_len·length − w_cross·same_layer_crossings`
(higher is better; same-layer crossings are hard-penalized — they must stay 0).

## How to train it (when you have GPU)
- **State**: board image (64×64×3) + per-layer congestion map + remaining-net mask.
- **Action**: `(net, layer)` each step; episode ends when all nets are assigned.
- **Reward**: `routing_reward` at terminal (or shape it per-step by Δlayers/Δvias).
- **Boards**: sample `ChallengeSpec` (vary `num_traces`, `n_gaps`, `board_size`, `seed`,
  `placement`) so the policy generalizes across difficulty — use `layer_budget.py` to
  see the difficulty surface.
- **Validate** a policy or heuristic on CPU first with
  `envs/formulations.py:evaluate_ordered_layer(board, placed, layer_of)` — no training
  loop needed.
