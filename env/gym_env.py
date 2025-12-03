import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .params import PulleyParams
from .pulley_env import PulleyEnv

class PulleyEnvGym(gym.Env):
    """
    Adapter that wraps existing simulator with the Gymnasium API.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, dt=0.001, max_steps=10000, render_mode=None, seed=None, action_repeat=10, max_action_diff=0.040):
        super().__init__()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.action_repeat = int(action_repeat) # used to store the number of steps the agent should hold the action
        self.sim = PulleyEnv(PulleyParams(), dt=self.dt, max_steps=self.max_steps) # create sim environment
        self.max_action_diff = max_action_diff
        self.u_max = self.sim.params.tau_max
        self.pulley_radius = self.sim.params.r1
        self.u_coact_max = (self.u_max - self.max_action_diff/2)*2/self.pulley_radius
        self.u_coact_min = (0 + self.max_action_diff/2)*2/self.pulley_radius
        self.render_mode = render_mode
        self.step_count = 0
        self.env_steps = 0

        # Goal config
        self.include_goal_flag = True
        self.theta_goal = 0
        self.theta_goal_init_range = (-np.pi/2, np.pi/2) # randomization goal position range for theta
        self.coact_goal = 0
        self.coact_goal_init_range = (self.u_coact_min, self.u_coact_max)
        self.goal_dim = 3 # since we are using cos theta, sin theta to encode the goal position
        self.random_theta_flag = True
        self.theta_init_range = (-np.pi/2, np.pi/2)
        self.random_dtheta_flag = True
        self.dtheta_init_range = (-np.deg2rad(45.0), np.deg2rad(45.0))
        
        # Variables on whether or random disturbance force should be enabled
        self.randomize_force_flag = False
        self.force_range = (-1.0, 1.0)

        # Tuning termination conditions for the environment
        self.success_streak = 0
        self.success_band = dict(angle=np.deg2rad(1.0), vel=np.deg2rad(.1), hold_steps=200, coact_force=0.1)
        self.safety_limits = dict(angle=np.deg2rad(110.0), vel=np.deg2rad(1080.0))
        
        # Making the velocity a soft termination condition and additionally adding a penalty as the limit is approached
        self.vel_soft_margin_ratio = 2/3                   # start penalizing at 80% of limit
        self.vel_violate_patience = 50      # must exceed limit N consecutive steps (set to 100 to allow agent 10 actions to fix vel)
        self.w_vel_safety = 100.0                           # hinge penalty weight
        # runtime counters
        self._vel_violate_steps = 0

        # state/observation space (for our case state space = observation space) 
        base_obs_dim = 6  # [theta, dtheta, phi1, dphi1, phi2, dphi2]
        obs_dim = base_obs_dim + self.goal_dim
        obs_hi = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_hi, high=obs_hi, dtype=np.float32)
        
        # configuring action space
        low = np.array([-self.max_action_diff], dtype=np.float32)
        high = np.array([self.max_action_diff], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

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
            dtheta = rng.uniform(*self.dtheta_init_range)
        else:
            dtheta = 0
        x = base_obs.astype(np.float32, copy=True)
        x[:] = 0.0
        x[0] = np.float32(theta)
        x[1] = np.float32(dtheta)
        x[2] = -np.float32(theta)
        x[4] = np.float32(theta)
        self.sim._x = x

        # random external force
        if self.randomize_force_flag:
            self.sim.params.F = float(rng.uniform(*self.force_range))
        else:
            self.sim.params.F = 0.0

        # sample goal angle θ*
        self.theta_goal = float(rng.uniform(*self.theta_goal_init_range))

        # sample goal coactivation force
        self.coact_goal = float(rng.uniform(*self.coact_goal_init_range))

        self.step_count = 0
        self.env_steps = 0
        self.success_streak = 0
        self._vel_violate_steps = 0

        obs = self._observe()
        info = {"t": 0.0, "theta_goal": self.theta_goal, "coact_goal": self.coact_goal,"F": float(self.sim.params.F)}
        return obs, info
    
    def _map_reparam_to_torques(self, a: np.ndarray) -> np.ndarray:
        """
        Map reparameterized action [u_coact (N), delta (Nm)] to torques [tau1, tau2] (Nm),
        enforcing bounds so both torques are within [0, u_max] and |tau1 - tau2| <= 2*|delta|.
        """
        delta = float(a[0])

        # Base torque from coactivation: tau1 + tau2 = u_coact * r1, select base such that tau1 = tau2
        base = 0.5 * self.coact_goal * float(self.pulley_radius)

        tau1 = base + delta/2
        tau2 = base - delta/2

        return np.array([tau1, tau2], dtype=np.float32)

    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        total_reward = 0.0
        terminated_flag = False
        truncated_flag = False
        reason = None
        info = {}
        
        action = self._map_reparam_to_torques(action)

        # Run N inner physics steps with a zero-order hold on 'action'
        for i in range(self.action_repeat):
            self.step_count += 1  # physics step counter
            
            *_, info = self.sim.step(action)
            self.curr_action = action
            obs = self._observe()
            if bool((obs > 1e6).any() | (obs < -1e6).any()):
                print(obs)
            obs = np.clip(obs, -1e6, 1e6)

            if not np.isfinite(obs).all():
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                obs = np.zeros(11)
                # give a big negative and terminate in the event of a simulation error
                r, terms = -1e5, {"e2":0.0,"dq2":0.0,"u2":0.0,"du2":0.0}
                terminated_flag, reason = True, "simulation_error"
                if not np.isfinite(action).all():
                    print(action)
                    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    self.curr_action = action
            else:
                # reward PER physics step (sum across hold)
                r, terms = self._reward(obs, action)

                terminated_flag, reason = self._terminated(obs, action)

            total_reward += r

            # Penalty for velocity being close to limit
            total_reward -= self._vel_safety_penalty(obs[1])

            # time-limit and early exit
            truncated_flag = self.step_count >= self.max_steps
            if terminated_flag or truncated_flag:
                break 

        # agent step book-keeping
        self.env_steps += 1
        # time reported to the agent can be macro time:
        t = float(self.step_count*self.dt)

        if terminated_flag and reason == "success":
            print("success")
            total_reward += 2000
        elif terminated_flag and (reason == "angle_limit"):
            # print("angle_limit")
            total_reward -= 10000
        elif terminated_flag and (reason == "velocity_limit"):
            # print("velocity_limit")
            total_reward -= 10000

        info = {
            "t": t,
            "theta_goal": self.theta_goal,
            "coact_goal": self.coact_goal,
            "reward_terms": terms,
            "success_streak": self.success_streak,
            "F": float(self.sim.params.F),
        }
        if terminated_flag or truncated_flag:
            print(reason)
            info["is_success"] = bool(terminated_flag and (reason == "success"))
            info["termination_reason"] = reason if terminated_flag else "time_limit"

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
        # Encode goal angle as [cos θ*, sin θ*, e_theta, coact_force, e_coact]
        e_theta = self._wrap_pi(self.sim._obs()[0] - self.theta_goal)
        return np.array([np.cos(self.theta_goal), np.sin(self.theta_goal), e_theta], dtype=np.float32)

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

        theta, dtheta, e_theta = float(obs[0]), float(obs[1]), float(obs[8])

        # weights (tune as needed)
        w_e  = 100         # angle error

        cost = w_e*e_theta*e_theta

        reward = -cost
        
        terms = {"e2": e_theta*e_theta}
        
        return reward, terms

    def _terminated(self, obs, action):
        theta, dtheta = float(obs[0]), float(obs[1])
        
        # Success check
        if self._theta_success(obs) and self._coact_success(action):
            self.success_streak += 1
            if self.success_streak >= self.success_band["hold_steps"]:
                return True, "success"
        else:
            self.success_streak = 0
        
        # Angle limit check
        if abs(self._wrap_pi(theta)) > self.safety_limits["angle"]:
            return True, "angle_limit"
        
        # Soft velocity check
        if abs(dtheta) > self.safety_limits["vel"]:
            self._vel_violate_steps += 1
        else:
            self._vel_violate_steps = 0
        if self._vel_violate_steps >= self.vel_violate_patience:    
            return True, "velocity_limit"
        return False, None
    
    def _theta_success(self, obs):
        theta, dtheta = float(obs[0]), float(obs[1])
        e_theta = self._wrap_pi(theta - self.theta_goal)
        return (abs(e_theta) < self.success_band["angle"])
    
    def _coact_success(self, action):
        e_coact = self.coact_goal - self._get_coact(action)
        return (abs(e_coact) < self.success_band["coact_force"])
    
    def _vel_safety_penalty(self, dtheta: float) -> float:
        """
        Quadratic hinge penalty that starts at margin = ratio * vel_limit and
        ramps up smoothly to the hard limit.
        """
        limit = float(self.safety_limits["vel"])
        margin = float(self.vel_soft_margin_ratio) * limit

        # Setting a limit to how high the velocity can be for the calculation of the penalty
        # ad = abs(float(dtheta))
        ad = min(abs(float(dtheta)), limit * 5)
        
        if ad <= margin:
            return 0.0
        # normalize how far we are between margin and limit (0..1..+)
        denom = max(1e-6, limit - margin) # get the max just in case, to prevent nans
        x = (ad - margin) / denom
        return self.w_vel_safety * (x ** 2)
