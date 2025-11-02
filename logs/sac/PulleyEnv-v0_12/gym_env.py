# Code/env/gym_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Pull your pieces in
from .params import PulleyParams
from .pulley_env import PulleyEnv

class PulleyEnvGym(gym.Env):
    """
    Adapter that wraps your existing simulator with the Gymnasium API.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, dt=0.001, max_steps=10000, render_mode=None, seed=None, action_repeat=10):
        super().__init__()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.action_repeat = int(action_repeat) # used to store the number of steps the agent should hold the action
        self.sim = PulleyEnv(PulleyParams(), dt=self.dt, max_steps=self.max_steps) # create sim environment
        self.u_max = self.sim.params.tau_max
        self.pulley_radius = self.sim.params.r1
        self.render_mode = render_mode
        self.step_count = 0
        self.env_steps = 0
        self.prev_e_theta = 0.0
        self.prev_e_coact = None

        # Goal config
        self.include_goal_flag = True
        self.theta_goal = 0
        self.theta_goal_init_range = (-np.pi/2, np.pi/2) # randomization goal position range for theta
        self.include_coact_goal_flag = True
        self.coact_goal = 0
        self.coact_goal_init_range = (0,self.u_max*2/self.pulley_radius)
        self.goal_dim = 5 # since we are using cos theta, sin theta to encode the goal position
        self.random_theta_flag = True
        self.theta_init_range = (-np.pi/2, np.pi/2)
        self.random_dtheta_flag = True
        self.dtheta_init_std = np.deg2rad(10.0) # TESTING->enable if we decide to also randomize the speed of theta
        self._success_streak = 0
        self.success_band = dict(angle=np.deg2rad(2.0), vel=np.deg2rad(.1), hold_steps=400, coact_force=0.3)
        self.safety_limits = dict(angle=np.deg2rad(100.0), vel=np.deg2rad(7200.0))
        # Variables on whether or random disturbance force should be enabled
        self.randomize_force_flag = False
        self.force_range = (-1.0, 1.0)

        # state/observation space (for our case state space = observation space) 
        base_obs_dim = 6  # [theta, dtheta, phi1, dphi1, phi2, dphi2]
        obs_dim = base_obs_dim + self.goal_dim
        obs_hi = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_hi, high=obs_hi, dtype=np.float32)
        # action space
        self.action_space = spaces.Box(
            low=0, high=self.u_max, shape=(2,), dtype=np.float32
        ) # shape of 2 for tau1 and tau2
        
        # TESTING->variable below is for action smoothness
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.curr_action = np.zeros(2, dtype=np.float32)

        # TESTING->maybe use this if we want to train the model to see if it can track a desired trajectory
        # self.theta_ref_fn = lambda t: 0.0   # e.g., lambda t: 0.5*np.sin(0.5*t)
        # self.success_band = dict(angle=np.deg2rad(3.0), vel=np.deg2rad(10.0), hold_steps=25)
        # self.prev_action = np.zeros(2, dtype=np.float32)
        # self.step_count = 0

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            base_obs, info = self.sim.reset(seed=seed)
            rng = np.random.default_rng(seed)
        else:
            base_obs, info = self.sim.reset()
            rng = np.random.default_rng()

        # randomize start θ, dθ
        if self.random_theta_flag:
            theta = rng.uniform(*self.theta_init_range)
        else:
            theta = 0
        if self.random_dtheta_flag:
            dtheta = rng.normal(0.0, self.dtheta_init_std)
        else:
            dtheta = 0
        x = base_obs.astype(np.float32, copy=True)
        x[0] = np.float32(theta)
        x[1] = np.float32(dtheta)
        x[2:] = 0.0
        self.sim._x = x

        # random external force
        if self.randomize_force_flag:
            self.sim.params.F = float(rng.uniform(*self.force_range))
        else:
            self.sim.params.F = 0.0

        # sample goal angle θ*
        self.theta_goal = float(rng.uniform(*self.theta_goal_init_range))
        self.prev_e_theta = self._wrap_pi(self.sim._obs()[0] - self.theta_goal)
        self.prev_e_coact = None

        # sample goal coactivation force
        self.coact_goal = float(rng.uniform(*self.coact_goal_init_range))

        self.step_count = 0
        self.env_steps = 0
        self.prev_action[:] = 0.0
        self._success_streak = 0

        obs = self._observe()
        info = {"t": 0.0, "theta_goal": self.theta_goal, "coact_goal": self.coact_goal,"F": float(self.sim.params.F)}
        return obs, info
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # DEBUGGING->printing out actions
        # print(action)
        
        total_reward = 0.0
        terminated_flag = False
        truncated_flag = False
        reason = None
        info = {}

        # Run N inner physics steps with a zero-order hold on 'action'
        for i in range(self.action_repeat):
            self.step_count += 1  # physics step counter
            
            *_, info = self.sim.step(action)
            self.curr_action = action
            obs = self._observe()

            if not np.isfinite(obs).all():
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                # give a big negative and terminate
                r, terms = -1e7, {"e2":0.0,"dq2":0.0,"u2":0.0,"du2":0.0}
                terminated_flag, reason = True, "simulation_error"
                print("simulation_error")
                if not np.isfinite(action).all():
                    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            else:
                # reward PER physics step (sum across hold)
                r, terms = self._reward(obs, action)

                terminated_flag, reason = self._terminated(obs, action)

            total_reward += r

            # time-limit and early exit
            truncated_flag = self.step_count >= self.max_steps
            if terminated_flag or truncated_flag:
                break 

        # agent step book-keeping
        self.env_steps += 1
        # time reported to the agent can be macro time:
        t = float(self.env_steps * self.action_repeat * self.dt)

        # print(self.env_steps)
        # print(self.step_count)

        if terminated_flag and reason == "success":
            print("success")
            total_reward += 40000.0
        elif terminated_flag and (reason == "angle_limit"):
            print("angle_limit")
            total_reward -= 150000.0
        elif terminated_flag and (reason == "velocity_limit"):
            print("velocity_limit")
            total_reward -= 150000.0

        info = {
            "t": t,
            "theta_goal": self.theta_goal,
            "coact_goal": self.coact_goal,
            "reward_terms": terms,
            "success_streak": self._success_streak,
            "F": float(self.sim.params.F),
        }
        if terminated_flag or truncated_flag:
            info["is_success"] = bool(terminated_flag and (reason == "success"))
            info["termination_reason"] = reason if terminated_flag else "time_limit"
            print(info["termination_reason"])

        return obs.astype(np.float32), float(total_reward), bool(terminated_flag), bool(truncated_flag), info

    def render(self):
        # You can forward to ui.render_matplotlib if you want visuals from Gym
        pass

    def close(self):
        pass

    # ------------- helpers specific to your sim -------------

    def _wrap_pi(self, x):
        """
        Wrap to (-pi, pi]
        """
        return (x + np.pi) % (2*np.pi) - np.pi
    
    def _goal_vec(self):
        # Encode goal angle as [cos θ*, sin θ*, coact_force]
        e_theta = self._wrap_pi(self.sim._obs()[0] - self.theta_goal)
        u_c = (self.curr_action[0] + self.curr_action[1]) / self.pulley_radius
        e_coact = u_c - self.coact_goal
        return np.array([np.cos(self.theta_goal), np.sin(self.theta_goal), e_theta, 
                         self.coact_goal, e_coact], dtype=np.float32)

    def _get_coact(self, action):
        return (action[0]+action[1])/self.pulley_radius
    
    def _observe(self):
        """
        Return observation vector as float32.
        Example: [theta, dtheta, phi1, dphi1, phi2, dphi2]
        """
        if self.include_goal_flag:
            return np.concatenate([self.sim._obs(), self._goal_vec()], dtype=np.float32)
        else:
            return self.sim._obs()

    # ------------- task-related functions -------------

    def _reward(self, obs, action):

        theta, dtheta, e_theta, e_coact = float(obs[0]), float(obs[1]), float(obs[8]), float(obs[10])

        # weights (tune as needed)
        w_e  = 9e-1         # angle error
        w_d  = 1e-1        # small velocity penalty
        w_u  = 100e-1         # action effort
        w_du = 200e-1         # action smoothness
        w_coact = 9e-1         # coactivation error

        u2  = float(np.dot(action, action))
        du2 = float(np.dot(action - self.prev_action, action - self.prev_action))
        cost = w_e*e_theta*e_theta + w_d*dtheta*dtheta + w_u*u2 + w_du*du2 + w_coact*e_coact*e_coact

        # --- progress bonus (shaping) ---
        theta_progress_bonus = 5.0 if abs(self.prev_e_theta) > abs(e_theta) else 0.0
        if self.prev_e_coact is not None:
            coact_progress_bonus = 10.0 if abs(self.prev_e_coact) > abs(e_coact) else 0.0
        else:
            coact_progress_bonus = 0
        reward = -cost + theta_progress_bonus + coact_progress_bonus

        # book-keeping for next step
        self.prev_action = action.astype(np.float32, copy=True)
        self.prev_e_theta = e_theta
        self.prev_e_coact = e_coact
        
        terms = {"e2": e_theta*e_theta, "dq2": dtheta*dtheta, "u2": u2, "du2": du2, "e_coact2": e_coact*e_coact}

        # TESTING->giving small reward if theta is within the success range
        if self._theta_success(obs):
            reward += 20
            if self._coact_success(action):
                reward += 40
        
        return reward, terms

    def _terminated(self, obs, action):
        theta, dtheta = float(obs[0]), float(obs[1])
        
        if self._theta_success(obs) and self._coact_success(action):
            # print("incrementing success streak")
            self._success_streak += 1
            if self._success_streak >= self.success_band["hold_steps"]:
                return True, "success"
        else:
            self._success_streak = 0
        
        if abs(self._wrap_pi(theta)) > self.safety_limits["angle"]:
            # DEBUGGING->verifying whether or not angle safety limit has been exceeded
            # print("angle safety limit exceeded")
            return True, "angle_limit"
        if abs(dtheta) > self.safety_limits["vel"]:
            # DEBUGGING->verifying whether or not velocity safety limit has been exceeded
            # print("velocity safety limit exceeded")
            return True, "velocity_limit"
        return False, None
    
    def _theta_success(self, obs):
        theta, dtheta = float(obs[0]), float(obs[1])
        e_theta = self._wrap_pi(theta - self.theta_goal)
        return (abs(e_theta) < self.success_band["angle"])
    
    def _coact_success(self, action):
        e_coact = self.coact_goal - self._get_coact(action)
        return (abs(e_coact) < self.success_band["coact_force"])
