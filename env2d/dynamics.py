import numpy as np
from .params import PulleyParams

STATE_SIZE = 10  # [theta1, dtheta1, theta2, dtheta2, phi1, dphi1, phi2, dphi2, phi3, dphi3]

class Inputs:
    def __init__(self):  # step-wise controls (actions)
        self.tau1 = 0.0
        self.tau2 = 0.0
        self.tau3 = 0.0

def make_dynamics(params: PulleyParams, inputs: Inputs):

    def f(x):
        theta1, dtheta1, theta2, dtheta2, phi1, dphi1, phi2, dphi2, phi3, dphi3 = x
        r1 = params.r1; r2 = params.r2; r3 = params.r3; r4 = params.r4; r5 = params.r5; 
        l2 = params.l2
        s10  = params.s10; s20 = params.s20; s30 = params.s30
        I1   = params.I1; I2 = params.I2; I3 = params.I3; I4 = params.get_I4(theta2); I5 = params.I5
        c1   = params.c1; c2 = params.c2; c3 = params.c3; c4 = params.c4; c5 = params.c5
        tau1 = inputs.tau1; tau2 = inputs.tau2; tau3 = inputs.tau3; F = params.F

        s1 = s10 + r1*phi1 + r5*theta2 + r4*theta1
        s2 = s20 + r2*phi2 - r5*theta2 + r4*theta1
        s3 = s30 + r3*phi3 - r4*theta1

        # debug
        print(params.k_eff(s1), params.k_eff(s2), params.k_eff(s3))
        # print(params.k_eff(s1)*s1, params.k_eff(s2)*s2, params.k_eff(s3)*s3)

        # Spring tensions (has to be >0)
        S1 = max(0,params.k_eff(s1)*s1); S2 = max(0, params.k_eff(s2)*s2); S3 = max(0, params.k_eff(s3)*s3)

        ddphi1 = (tau1 - c1*dphi1 - r1*S1) / I1
        ddphi2 = (tau2 - c2*dphi2 - r2*S2) / I2
        ddphi3 = (tau3 - c3*dphi3 - r3*S3) / I3
        ddtheta1 = (r4*(S3 - S2 - S1) - c4*dtheta1) / I4
        # print(ddtheta1, dtheta1)
        ddtheta2 = (r5*(S2 - S1) + F*l2 - c5*dtheta2) / I5

        return np.array([dtheta1, ddtheta1, dtheta2, ddtheta2, dphi1, ddphi1, dphi2, ddphi2, dphi3, ddphi3], dtype=float)
    return f
