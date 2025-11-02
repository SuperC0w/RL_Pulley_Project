import os
import numpy as np
import argparse
from collections import defaultdict

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.gym_env import PulleyEnvGym


def make_env(seed=123, dt=0.001, max_steps=2500):
    def _thunk():
        return PulleyEnvGym(dt=dt, max_steps=max_steps, seed=seed)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/SAC/best/best_model.zip",
                        help="Path to SAC .zip (best or final)")
    parser.add_argument("--vecnorm", type=str, default="./models/SAC/best/best_vecnorm.pkl",
                        help="Path to VecNormalize .pkl saved alongside the model")
    parser.add_argument("--episodes", type=int, default=5, help="How many eval episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    args = parser.parse_args()

    # ----- Build 1-env VecEnv and (optionally) load VecNormalize stats -----
    venv = DummyVecEnv([make_env()])
    if os.path.isfile(args.vecnorm):
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False  # freeze stats at eval time
    else:
        print(f"[warn] VecNormalize file not found: {args.vecnorm}. Proceeding without it.")

    # ----- Load policy -----
    model = SAC.load(args.model, device="cpu")

    # ----- Rollout and collect reward terms -----
    obs = venv.reset()
    ep = 0
    step = 0

    # buffers
    per_step = []  # raw step-by-step dicts
    agg = defaultdict(list)  # per-episode aggregates

    while ep < args.episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, infos = venv.step(action)

        # infos is a list (length = n_envs). We have n_envs=1, so take infos[0].
        info = infos[0]
        rt = info.get("reward_terms", None)  # {'e2', 'dq2', 'u2', 'du2', 'e_coact2'} in your env
        if rt is not None:
            row = dict(step=step, ep=ep, **{k: float(rt[k]) for k in rt})
            per_step.append(row)

        if done[0]:
            # Episode end: compute simple aggregates for this episode
            ep_rows = [r for r in per_step if r["ep"] == ep]
            if ep_rows:
                for k in ("e2", "dq2", "u2", "du2", "e_coact2"):
                    vals = [r[k] for r in ep_rows if k in r]
                    if vals:
                        agg[f"{k}_mean"].append(np.mean(vals))
                        agg[f"{k}_max"].append(np.max(vals))
                        agg[f"{k}_last"].append(vals[-1])

                print(f"\n[episode {ep}] steps={len(ep_rows)} "
                      + " ".join([f"{k}: {np.mean(v):.4g}" for k, v in agg.items() if k.endswith('_mean')]))
            ep += 1
            venv.reset()

        step += 1

    # ----- Final summary -----
    if agg:
        print("\n=== Summary over episodes ===")
        for key in sorted(agg.keys()):
            print(f"{key}: {np.mean(agg[key]):.6g}")

    print("\nDone.")


if __name__ == "__main__":
    main()
