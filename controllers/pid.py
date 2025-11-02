import numpy as np

class PID:
    def __init__(
        self,
        kp=5.0, ki=0.0, kd=0.001,
        kp_sat = 0.0025,
        kd_sat = 0.001,
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
        tau1 = coact_goal*self.radius/2
        tau2 = coact_goal*self.radius/2
        if e > 0:
            # print(min(abs(e)*self.kp, 0.005))
            # print(np.clip(self.kd * de, -self.kd_sat, self.kd_sat))
            # print(e,self.kd * de)
            tau2 += min(abs(e)*self.kp, self.kp_sat) - np.clip(self.kd * dtheta, -self.kd_sat, self.kd_sat)
        else:
            # print(min(abs(e)*self.kp, 0.005))
            # print(np.clip(self.kd * de, -self.kd_sat, self.kd_sat))
            # print(e,self.kd * de)
            tau1 += min(abs(e)*self.kp, self.kp_sat) + np.clip(self.kd * dtheta, -self.kd_sat, self.kd_sat)
            # tau1 += 0.005
        self.prev_e = e
        return np.array([tau1, tau2], float)
