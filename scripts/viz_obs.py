"""Visualize the 64x64x3 observation the agent sees, and run a quick env smoke check.

Channels:  R = obstacles / edges,  G = placed test points,  B = current trace + candidates.
Run from the repo root:  python scripts/viz_obs.py
Writes eval_results/obs_initial.png and obs_final.png. Needs matplotlib.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

import numpy as np
from envs.pcb_env import TPPlacementEnv


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path("eval_results")
    out.mkdir(exist_ok=True)
    env = TPPlacementEnv(num_traces=8, seed=0)
    obs, info = env.reset(seed=42)
    print(f"Obs {obs.shape} dtype={obs.dtype} range=[{obs.min()},{obs.max()}]  "
          f"action=Discrete({env.num_candidates})  traces={env.num_traces}")

    def save(o, name):
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(o); ax[0].set_title("Combined")
        ax[1].imshow(o[:, :, 0], cmap="Reds"); ax[1].set_title("R: obstacles/edges")
        ax[2].imshow(o[:, :, 1], cmap="Greens"); ax[2].set_title("G: placed TPs")
        ax[3].imshow(o[:, :, 2], cmap="Blues"); ax[3].set_title("B: current + candidates")
        for a in ax:
            a.axis("off")
        plt.savefig(out / name, dpi=100, bbox_inches="tight")
        plt.close()
        print("saved", out / name)

    save(obs, "obs_initial.png")
    total = 0.0
    step = 0
    for step in range(env.num_traces):
        valid = np.where(env.candidate_mask)[0]
        a = int(np.random.choice(valid)) if len(valid) else env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        total += r
        if done:
            break
    print(f"episode: {step + 1} steps, reward={total:.1f}, info={info}")
    save(obs, "obs_final.png")

    from gymnasium.utils.env_checker import check_env
    check_env(TPPlacementEnv(num_traces=8), skip_render_check=True)
    print("gymnasium env check: PASSED")


if __name__ == "__main__":
    main()
