import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from env.gym_env import PulleyEnvGym
import argparse

def make_env(seed):
    def _thunk():
        return PulleyEnvGym(dt=0.001, max_steps=10000, seed=seed)
    return _thunk

def build_env(n_envs=8, seed_base=100):
    venv = DummyVecEnv([make_env(seed_base + i) for i in range(n_envs)])
    return VecMonitor(venv)

class SuccessRateCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.window_eps = 0
        self.window_success = 0
        self.steps_since_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            self.window_eps += 1
            if infos[i].get("is_success", False):
                self.window_success += 1

        self.steps_since_log += 1
        if self.steps_since_log >= self.log_every and self.window_eps > 0:
            rate = self.window_success / self.window_eps
            self.logger.record("rollout/success_rate", rate)  # shows up in TensorBoard
            # reset window
            self.steps_since_log = 0
            self.window_eps = 0
            self.window_success = 0
        return True
    
def build_callbacks(n_envs):
    eval_env = build_env(1, seed_base=999)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/PPO/best",
        log_path="./models/PPO/eval",
        eval_freq=25_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path="./models/PPO/ckpts",
        name_prefix="ppo",
        save_replay_buffer=True,   # helpful for true resume
    )
    success_cb = SuccessRateCallback(log_every=5000)
    return [eval_cb, ckpt_cb, success_cb]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--resume-from", type=str, default="")   # path to .zip (optional)
    parser.add_argument("--replay", type=str, default="")        # path to *_rb.pkl (optional)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "auto"
    venv = build_env(args.n_envs)
    callbacks = build_callbacks(args.n_envs)

    if args.resume_from:
        # ---- Resume path ----
        model = PPO.load(args.resume_from, device=device)
        model.set_env(venv)
        if args.replay:
            model.load_replay_buffer(args.replay)
        model.learn(
            total_timesteps=args.total_steps,
            reset_num_timesteps=False,        # <-- continue global step counter
            tb_log_name="PPO_resume",
            callback=callbacks,
        )
    else:
        # ---- Fresh training ----
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            tensorboard_log="./tb",
            device='cpu',
            # Some sensible defaults
            n_steps=2048 // args.n_envs,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5
        )
        model.learn(
            total_timesteps=args.total_steps,
            tb_log_name="PPO",
            callback=callbacks,
        )

    model.save("./models/PPO/final/ppo_pulley")
