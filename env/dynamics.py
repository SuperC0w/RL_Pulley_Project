import numpy as np
from .params import PulleyParams

STATE_SIZE = 6  # [theta, dtheta, phi1, dphi1, phi2, dphi2]

class Inputs:
    def __init__(self):  # step-wise controls (actions)
        self.tau1 = 0.0
        self.tau2 = 0.0

def make_dynamics(params: PulleyParams, inputs: Inputs):

    def f(x):
        theta, dtheta, phi1, dphi1, phi2, dphi2 = x
        r1 = params.r1; r2 = params.r2; r3 = params.r3; l = params.l
        # spring stiffnesses
        k1   = params.k1; k2 = params.k2
        alpha1 = params.alpha1; alpha2 = params.alpha2 
        s10  = params.s10; s20 = params.s20
        I1   = params.I1; I2 = params.I2; I3 = params.I3
        c1   = params.c1; c2 = params.c2; c3 = params.c3
        tau1 = inputs.tau1; tau2 = inputs.tau2; F = params.F

        s1 = s10 + r3*theta + r1*phi1
        s2 = s20 - r3*theta + r2*phi2
        # spring tensions (ensured that is >0)
        S1 = max(0, params.k_eff(k1,s1,alpha1)*s1); S2 = max(0,params.k_eff(k2,s2,alpha2)*s2)
        ddphi1 = (tau1 - c1*dphi1 - r1*S1) / I1
        ddphi2 = (tau2 - c2*dphi2 - r2*S2) / I2
        ddtheta = (r3*(S2 - S1) + F*l - c3*dtheta) / I3

        return np.array([dtheta, ddtheta, dphi1, ddphi1, dphi2, ddphi2], dtype=float)
    return f
