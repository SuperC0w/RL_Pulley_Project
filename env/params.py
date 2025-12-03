from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class PulleyParams:
    density = 2700        # kg/m^3 (density of the pulley material)
    h1: float = 0.01     # m (pulley 1 height)
    h2: float = 0.01     # m (pulley 2 height)
    h3: float = 0.01     # m (pulley 3 height)
    r1: float = 0.03     # m (pulley 1 radius)
    r2: float = 0.03     # m (pulley 2 radius)
    r3: float = 0.03     # m (pulley 3 radius) 
    l: float = 0.1      # m (pulley 3 link length)
    w: float = 0.01     # m (pulley 3 link width)
    t: float = 0.005
    m1: float = np.pi*r1**2*h1*density         # kg (mass of pulley 1)
    m2: float = np.pi*r2**2*h2*density         # kg (mass of pulley 2)
    m3: float = np.pi*r3**2*h3*density         # kg (mass of pulley 3)
    m_link: float = l*w*t*density                # kg (mass of link)

    I1: float = 11/10000000 + m1*r1**2/2     # kg*m^2 (pulley 1 inertia, first term is the motor inertia)
    I2: float = 11/10000000 + m2*r2**2/2     # kg*m^2 (pulley 2 inertia, first term is the motor inertia)
    I3: float = m3*r3**2/2 + (m_link*(l**2+w**2)/12 + m_link*(l/2)**2)      # kg*m^2 (pulley 3 inertia, first term is for pulley and second term is for link)
    c1: float = 2.53e-4   # N*m*s (pulley 1 damping)
    c2: float = 2.53e-4   # N*m*s (pulley 2 damping)
    c3: float = 6.9e-8    # N*m*s (pulley 3 damping)

    k1: float = 100.0   # N/m (spring 1 stiffness)
    k2: float = 100.0   # N/m (spring 2 stiffness)
    alpha1: float = 8e5
    alpha2: float = 8e5
    # alpha1 = 16e5
    # alpha2 = 16e5
    # alpha1 = 1e4
    # alpha2 = 1e4
    s10: float = 0       # m (spring pre-extension)
    s20: float = 0       # m (spring pre-extension)
    F: float = 0         # N (applied force at the end of the link)

    # Local function to get the stiffness
    def k_eff(self, k0, x, alpha):
        return k0 * (1.0 + alpha * x * x)

    # action limits 
    tau_max: float = 0.129   # N*m (max torque magnitude that can be exerted)
