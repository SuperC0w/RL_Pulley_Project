import numpy as np
from .params import PulleyParams
from .dynamics import make_dynamics, Inputs, STATE_SIZE
from .integrators import rk4_step

class PulleyEnv:
    def __init__(self, params: PulleyParams, dt=0.003, max_steps=10_000, stepper=rk4_step, seed=None):
        self.params = params
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.stepper = stepper
        self.rng = np.random.default_rng(seed)

        self.inputs = Inputs()
        self._f = make_dynamics(self.params, self.inputs)
        self._x = None
        self._t = 0.0
        self._steps = 0

        self.baseline_force = params.F
        self.impulse_force = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._x = np.zeros(STATE_SIZE, dtype=np.float32)

        # TESTING->winding up motor pulleys when setting initial positions of theta 1 and theta 2
        # theta1 = np.pi/4
        # theta2 = np.pi/2
        # self._x[:] = 0.0
        # self._x[0] = np.float32(theta1)
        # self._x[2] = np.float32(theta2)
        # self._x[4] = np.float32(-theta1*self.params.r4/self.params.r1 - theta2*self.params.r5/self.params.r1)
        # self._x[6] = np.float32(-theta1*self.params.r4/self.params.r2 + theta2*self.params.r5/self.params.r2)
        # self._x[8] = np.float32(theta1*self.params.r4/self.params.r3)

        self._t = 0.0
        self._steps = 0
        obs = self._obs()
        info = {"t": self._t}
        return obs, info
    
    def trigger_impulse(self, F: float = 2):
        self.impulse_force = F

    def step(self, action):
        # action = [tau1, tau2, tau3]
        a = np.asarray(action, dtype=float).ravel()
        self.inputs.tau1, self.inputs.tau2, self.inputs.tau3 = (a[0], a[1], a[2])
        # clamp using params
        self.inputs.tau1 = float(np.clip(self.inputs.tau1, -self.params.tau_max1, self.params.tau_max1))
        self.inputs.tau2 = float(np.clip(self.inputs.tau2, -self.params.tau_max2, self.params.tau_max2))
        self.inputs.tau3 = float(np.clip(self.inputs.tau3, -self.params.tau_max3, self.params.tau_max3))

        if self.impulse_force is not None:
            self.params.F = self.impulse_force
        else:
            self.params.F = self.baseline_force

        self._x = self.stepper(self._f, self._x, self.dt)
        self._t += self.dt
        self._steps += 1

        self.impulse_force = None

        obs = self._obs()
        info = {"t": self._t}
        return obs, info

    def _obs(self): 
        return self._x.copy()
