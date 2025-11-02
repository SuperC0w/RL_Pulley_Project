import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from env.gym_env import PulleyEnvGym
import argparse

def make_env(seed):
    def _thunk():
        return PulleyEnvGym(dt=0.001, max_steps=2500, seed=seed)
    return _thunk

def build_env(n_envs=8, seed_base=100):
    venv = DummyVecEnv([make_env(seed_base + i) for i in range(n_envs)])
    venv = VecCheckNan(venv, raise_exception=True) 
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    return venv

class EvalCallbackSaveVecNorm(EvalCallback):
    """
    Wraps EvalCallback and, whenever a new best model is saved,
    also saves the current training VecNormalize object alongside it.
    """
    def __init__(self, *args, train_venv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_venv = train_venv
        self._prev_best = -float("inf")

    def _on_step(self) -> bool:
        ok = super()._on_step()
        # After evals, EvalCallback keeps last_mean_reward and best_mean_reward
        # If improved, persist VecNormalize next to best model
        if getattr(self, "last_mean_reward", None) is not None:
            if self.best_mean_reward is not None and self.best_mean_reward > self._prev_best:
                self._prev_best = self.best_mean_reward
                if isinstance(self.train_venv, VecNormalize) and self.best_model_save_path is not None:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.train_venv.save(os.path.join(self.best_model_save_path, "best_vecnorm.pkl"))
        return ok

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

def build_callbacks(n_envs, train_venv):
    eval_env = build_env(1, seed_base=999)
    eval_cb = EvalCallbackSaveVecNorm(
        eval_env,
        best_model_save_path="./models/SAC/best",
        log_path="./models/SAC/eval",
        eval_freq=100_000 // n_envs,
        n_eval_episodes=50,
        deterministic=True,
        train_venv=train_venv
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path="./models/SAC/ckpts",
        name_prefix="sac",
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
    callbacks = build_callbacks(args.n_envs, venv)

    # TESTING-> syncing normalization stats
    eval_cb = callbacks[0]  # your EvalCallback
    if isinstance(venv, VecNormalize):
        # copy running stats for observations (and returns, if used)
        eval_cb.eval_env.obs_rms = venv.obs_rms
        eval_cb.eval_env.ret_rms = venv.ret_rms
        # put eval env in eval mode (don’t update stats, don’t normalize rewards)
        eval_cb.eval_env.training = False
        eval_cb.eval_env.norm_reward = False

    print("EVAL eval_freq =", callbacks[0].eval_freq)  # assuming eval_cb is first

    if args.resume_from:
        # ---- Resume path ----
        model = SAC.load(args.resume_from, device=device)
        model.set_env(venv)
        if args.replay:
            model.load_replay_buffer(args.replay)
        model.learn(
            total_timesteps=args.total_steps,
            reset_num_timesteps=False,        # <-- continue global step counter
            tb_log_name="SAC_resume",
            callback=callbacks,
        )
    else:
        # ---- Fresh training ----
        model = SAC(
            "MlpPolicy",
            venv,
            device=device,
            verbose=1,
            tensorboard_log="./tb",
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto"
        )
        model.learn(
            total_timesteps=args.total_steps,
            tb_log_name="SAC",
            callback=callbacks,
        )

    if isinstance(venv, VecNormalize):
        venv.save("./models/SAC/final/final_vecnorm")
    model.save("./models/SAC/final/sac_pulley")

if __name__ == "__main__":
    main()
