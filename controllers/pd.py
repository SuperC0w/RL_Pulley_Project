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
        self.radius = float(radius)
        self.kp_sat = kp_sat
        self.kd_sat = kd_sat

        self.int_e = 0.0
        self.prev_e = 0.0

    @staticmethod
    def wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def reset(self):
        self.int_e = 0.0
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

class PD_2d:
    def __init__(
        self,
        kp1=0.006, kp2=0.002, 
        ki1=0.0, ki2=0.0,
        kd1=0.0004, kd2=0.001,
        kp1_sat=np.inf, kp2_sat=np.inf,
        kd1_sat=np.inf, kd2_sat=np.inf,
        dt=0.001,
    ):
        self.kp1, self.ki1, self.kd1 = float(kp1), float(ki1), float(kd1)
        self.kp2, self.ki2, self.kd2 = float(kp2), float(ki2), float(kd2)
        self.dt = float(dt)
        self.kp1_sat, self.kp2_sat = kp1_sat, kp2_sat
        self.kd1_sat, self.kd2_sat = kd1_sat, kd2_sat

        self.int_e = 0.0
        self.prev_e1 = 0.0
        self.prev_e2 = 0.0

    @staticmethod
    def wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def reset(self):
        self.int_e = 0.0
        self.prev_e1 = 0.0
        self.prev_e2 = 0.0

    def step(self, state, theta1_goal, theta2_goal, coact_goal):
        # TODO->get relevant states from state vector
        # angle error with wrap
        # print(theta)
        theta1 = state[0]
        theta2 = state[2]

        e1 = self.wrap_pi(theta1_goal - theta1)
        e2 = self.wrap_pi(theta2_goal - theta2)
        de1 = (e1 - self.prev_e1)/self.dt
        de2 = (e2 - self.prev_e2)/self.dt
        
        base = coact_goal

        kp1_term = np.clip(e1*self.kp1, -self.kp1_sat, self.kp1_sat)
        kp2_term = np.clip(e2*self.kp1, -self.kp2_sat, self.kp2_sat)
        kv1_term = np.clip(self.kd1*de1, -self.kd1_sat, self.kd1_sat)
        kv2_term = np.clip(self.kd2*de2, -self.kd2_sat, self.kd2_sat)
        delta1 = kp1_term + kv1_term
        delta2 = kp2_term + kv2_term

        # TESTING->enforcing limits to the magnitudes of delta 
        # delta1 = np.clip(delta1, -0.020, 0.020)
        # delta2 = np.clip(delta2, -0.010, 0.010)
        # print(delta1)
        # print(delta2)

        tau1 = base/2 - delta2/2 - delta1/4
        tau2 = base/2 + delta2/2 - delta1/4
        tau3 = base + delta1/2

        self.prev_e1 = e1
        self.prev_e2 = e2
        
        if tau1 < 0:
            print("tau1<0")
        elif tau2 < 0:
            print("tau2<0")
        elif tau3 < 0:
            print("tau3<0")
        return np.array([tau1, tau2, tau3], float)