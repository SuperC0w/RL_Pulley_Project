
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.gym_env import PulleyEnvGym


class PulleyEnvGymReparam(PulleyEnvGym):
    """
    Reparameterized action interface for the Pulley Gym env.

    - Policy outputs: [u_coact (N), delta (Nm)]
    - Internally mapped to torques [tau1, tau2] with a small, bounded
      difference so that |tau1 - tau2| <= max_action_diff.
    """

    def __init__(
        self,
        dt: float = 0.001,
        max_steps: int = 10000,
        render_mode: str | None = None,
        seed: int | None = None,
        action_repeat: int = 10,
        delta_max: float | None = None,
    ) -> None:
        super().__init__(dt=dt, max_steps=max_steps, render_mode=render_mode, seed=seed, action_repeat=action_repeat)

        # Bound on |tau1 - tau2|. In the base env, max_action_diff is the hinge
        # threshold for pairwise difference. We set |delta| <= max_action_diff/2
        # so that |tau1 - tau2| = 2|delta| <= max_action_diff.
        base_max_diff = float(getattr(self, "max_action_diff", 0.01))
        self.delta_max: float = float(delta_max) if delta_max is not None else (base_max_diff / 2.0)

        # Upper bound on coactivation force (N), derived from torque limits
        # u_c = (tau1 + tau2) / r1, with tau_i in [0, u_max]
        self.u_coact_max: float = float(self.u_max * 2.0 / self.pulley_radius)

        # Replace action space: [u_coact (N), delta (Nm)]
        low = np.array([0.0, -self.delta_max], dtype=np.float32)
        high = np.array([self.u_coact_max, self.delta_max], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ---------- mapping ----------
    def _map_reparam_to_torques(self, a: np.ndarray) -> np.ndarray:
        """
        Map reparameterized action [u_coact (N), delta (Nm)] to torques [tau1, tau2] (Nm),
        enforcing bounds so both torques are within [0, u_max] and |tau1 - tau2| <= 2*|delta|.
        """
        u_coact = float(a[0])
        delta = float(a[1])

        # Enforce delta bound for guaranteed pairwise difference
        delta = float(np.clip(delta, -self.delta_max, self.delta_max))

        # Base torque from coactivation: tau1 + tau2 = u_coact * r1
        base = 0.5 * u_coact * float(self.pulley_radius)

        # Ensure no saturation of tau1/tau2 by constraining base to [|delta|, u_max - |delta|]
        # This guarantees tau1, tau2 ∈ [0, u_max] with the chosen delta.
        bound = abs(delta)
        if self.u_max - bound < 0.0:
            # Extremely tight delta vs u_max (should not happen with typical values),
            # fall back to zero torques safely.
            return np.zeros(2, dtype=np.float32)
        base = float(np.clip(base, bound, self.u_max - bound))

        tau1 = base + delta
        tau2 = base - delta
        # Final safety clip (no-ops if base bound above was satisfied)
        tau1 = float(np.clip(tau1, 0.0, self.u_max))
        tau2 = float(np.clip(tau2, 0.0, self.u_max))
        return np.array([tau1, tau2], dtype=np.float32)

    # ---------- Gym API override ----------
    def step(self, action):
        # Interpret incoming action as [u_coact (N), delta (Nm)]
        a = np.asarray(action, dtype=np.float32)
        # Clip in the reparameterized space
        a = np.clip(a, self.action_space.low, self.action_space.high)

        total_reward = 0.0
        terminated_flag = False
        truncated_flag = False
        reason = None
        info = {}

        # Zero-order hold for 'action_repeat' inner physics steps
        tau_action = self._map_reparam_to_torques(a)

        for _ in range(self.action_repeat):
            self.step_count += 1

            # Step low-level sim with torques
            *_, info = self.sim.step(tau_action)
            self.curr_action = tau_action  # keep torques for goal/metrics

            obs = self._observe()
            if bool((obs > 1e6).any() | (obs < -1e6).any()):
                print(obs)
            obs = np.clip(obs, -1e6, 1e6)

            if not np.isfinite(obs).all():
                # Simulation error fallback
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                obs = np.zeros(11)
                r, terms = -1e5, {"e2": 0.0, "dq2": 0.0, "u2": 0.0, "du2": 0.0}
                terminated_flag, reason = True, "simulation_error"
                if not np.isfinite(tau_action).all():
                    print(tau_action)
                    tau_action = np.nan_to_num(tau_action, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    self.curr_action = tau_action
            else:
                # Use torques in reward/termination
                r, terms = self._reward(obs, tau_action)
                terminated_flag, reason = self._terminated(obs, tau_action)

            total_reward += r
            # Velocity safety shaping
            total_reward -= self._vel_safety_penalty(obs[1])

            # Time limit or early exit
            truncated_flag = self.step_count >= self.max_steps
            if terminated_flag or truncated_flag:
                break

        # Agent step bookkeeping
        self.env_steps += 1
        t = float(self.step_count * self.dt)

        if terminated_flag and reason == "success":
            print("success")
            total_reward += 5e6
        elif terminated_flag and (reason == "angle_limit"):
            total_reward -= 100000.0
        elif terminated_flag and (reason == "velocity_limit"):
            total_reward -= 100000.0

        info = {
            "t": t,
            "theta_goal": self.theta_goal,
            "coact_goal": self.coact_goal,
            "reward_terms": terms,
            "success_streak": self._success_streak,
            "F": float(self.sim.params.F),
        }
        if terminated_flag or truncated_flag:
            print(reason)
            info["is_success"] = bool(terminated_flag and (reason == "success"))
            info["termination_reason"] = reason if terminated_flag else "time_limit"

        return (
            obs.astype(np.float32),
            float(total_reward),
            bool(terminated_flag),
            bool(truncated_flag),
            info,
        )


__all__ = ["PulleyEnvGymReparam"]

