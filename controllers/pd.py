import numpy as np

class PD_1d:
    def __init__(
        self,
        kp=0.002, ki=0.0, kd=0.0004,
        kp_sat = np.inf,
        kd_sat = np.inf,
        dt=0.001,
        radius=0.03
    ):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.dt = float(dt)
        self.u = np.zeros(2)
        self.radius = float(radius)
        self.kp_sat = kp_sat
        self.kd_sat = kd_sat

        self.int_e = 0.0
        self.d_est = 0.0   # filtered derivative of error
        self.prev_e = 0.0

    @staticmethod
    def wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def reset(self):
        self.int_e = 0.0
        self.d_est = 0.0
        self.prev_e = 0.0

    def step(self, theta, dtheta, theta_goal, coact_goal):
        # angle error with wrap
        # print(theta)
        e = self.wrap_pi(theta_goal - theta)
        de = (e - self.prev_e)/self.dt
        base = 0.5 * coact_goal * float(self.radius)
        kp_term = np.clip(e*self.kp, -self.kp_sat, self.kp_sat)
        kv_term = np.clip(self.kd*de, -self.kd_sat, self.kd_sat)
        delta = kp_term + kv_term
        tau1 = base - delta/2
        tau2 = base + delta/2
        self.prev_e = e
        return np.array([tau1, tau2], float)

# TODO->need to implement 2d controller
class PD_2d:
    def __init__(
        self,
        kp=0.002, ki=0.0, kd=0.0004,
        kp_sat = np.inf,
        kd_sat = np.inf,
        dt=0.001,
        radius=0.03
    ):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.dt = float(dt)
        self.u = np.zeros(2)
        self.radius = float(radius)
        self.kp_sat = kp_sat
        self.kd_sat = kd_sat

        self.int_e = 0.0
        self.d_est = 0.0   # filtered derivative of error
        self.prev_e = 0.0

    @staticmethod
    def wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def reset(self):
        self.int_e = 0.0
        self.d_est = 0.0
        self.prev_e = 0.0

    def step(self, theta, dtheta, theta_goal, coact_goal):
        # angle error with wrap
        # print(theta)
        e = self.wrap_pi(theta_goal - theta)
        de = (e - self.prev_e)/self.dt
        base = 0.5 * coact_goal * float(self.radius)
        kp_term = np.clip(e*self.kp, -self.kp_sat, self.kp_sat)
        kv_term = np.clip(self.kd*de, -self.kd_sat, self.kd_sat)
        delta = kp_term + kv_term
        tau1 = base - delta/2
        tau2 = base + delta/2
        self.prev_e = e
        return np.array([tau1, tau2], float)