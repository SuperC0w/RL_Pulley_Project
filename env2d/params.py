from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class PulleyParams:
    density = 1240        # kg/m^3 (density of the pulley material)
    infill = 0.20         # infill percentage
    w_t = 0.0008           # 3d printer wall thickness

    # Pulley heights (m)
    h1: float = 0.0255; h2: float = 0.0255; h3: float = 0.0255 ;h4: float = 0.008 ;h5: float = 0.023
    # Pulley radius (m)
    r1: float = 0.01175/2; r2: float = 0.01175/2; r3: float = 0.01175/2; r4: float = 0.033/2; r5: float = 0.033/2 
    # Link lengths (m)
    l1: float = 0.12; l2: float = 0.126
    # Link widths (m)
    w1: float = 0.04; w2: float = 0.04 
    # Link thickness (m)
    t: float = 0.023
    
    # Mass of pulleys (kg)
    m1: float = np.pi*(r1+0.0108)**2*h1*density         # kg (mass of pulley 1)
    m2: float = np.pi*(r2+0.0108)**2*h2*density         # kg (mass of pulley 2)
    m3: float = np.pi*(r3+0.0108)**2*h3*density         # kg (mass of pulley 3)
    m4: float = np.pi*r4**2*h4*density         # kg (mass of pulley 4)
    m5: float = np.pi*r5**2*h5*density         # kg (mass of pulley 5)
    # Mass of links (kg)
    m_link1: float = l1*w1*t*density - l1*(w1-2*w_t)*(t-2*w_t)*density + l1*(w1-2*w_t)*(t-2*w_t)*density*infill
    m_link2: float = l2*w2*t*density - l2*(w2-2*w_t)*(t-2*w_t)*density + l2*(w2-2*w_t)*(t-2*w_t)*density*infill

    # Inertias (kg*m^2)
    I1: float = 11/10000000 + m1*r1**2/2     # kg*m^2 (pulley 1 inertia, first term is the motor inertia)
    I2: float = 11/10000000 + m2*r2**2/2     # kg*m^2 (pulley 2 inertia, first term is the motor inertia)
    I3: float = 11/10000000 + m3*r3**2/2     # kg*m^2 (pulley 3 inertia, first term is the motor inertia)
    I5: float = m5*r5**2/2 + (m_link2*(l2**2+w2**2)/12 + m_link2*(l2/2)**2)      # kg*m^2 (pulley 5 inertia, first term is for pulley and second term is for link)
    def get_I4(self, theta):
        I4: float = (self.m4*self.r4**2/2 + (self.m_link1*(self.l1**2+self.w1**2)/12 + self.m_link1*(self.l1/2)**2)
                        + self.m5*self.r5**2/2 + self.m5*self.l1**2 + 
                        self.m_link2*((self.l2**2+self.w2**2)/12 + self.l1**2 + (self.l2/2)**2 
                        + 2*self.l1*self.l2/2*np.cos(theta)))
        return I4
    # Pulley damping
    c1: float = 2.53e-4   # N*m*s (pulley 1 damping)
    c2: float = 2.53e-4   # N*m*s (pulley 2 damping)
    c3: float = 2.53e-4   # N*m*s (pulley 3 damping)
    c4: float = 6.9e-8    # N*m*s (theta 1 damping)
    c5: float = 6.9e-8    # N*m*s (theta 2 damping)
    
    F1: float = 0         # N (applied force at the end of the first link)
    F2: float = 0        # N (applied force at the second link)

    # Spring pre-extension (m)
    s10: float = 0; s20: float = 0; s30: float = 0

    # Local functions to get the stiffness
    k1_1: float = 2.41030557e-03
    A_1: float = 4.76443842e+01
    p_1: float = 1.33420633e+00
    xmax_1: float = 1.18143022e+02
    def k_eff1(self, x):
        """
        Stiffness function obtained from plot_stiffness_function
        
        :param x: Spring extension
        """

        eps = 1e-6
        x *= 1000  # convert to mm
        denom = np.maximum((self.xmax_1 - x), eps)
        return self.k1_1 + self.A_1 * self.p_1 / denom ** (self.p_1 + 1) * 1000

    k1_2: float = 2.60343696e-03
    A_2: float = 1.14318518e+01
    p_2: float = 7.91426620e-01
    xmax_2: float = 1.31860228e+02
    def k_eff2(self, x):
        """
        Stiffness function obtained from plot_stiffness_function
        
        :param x: Spring extension
        """
        eps = 1e-6
        x *= 1000  # convert to mm
        denom = np.maximum((self.xmax_2 - x), eps)
        return self.k1_2 + self.A_2 * self.p_2 / denom ** (self.p_2 + 1) * 1000

    k1_3: float = 1.58356171e-03
    A_3: float = 1.15052582e+01
    p_3: float = 2.029421835e+00
    xmax_3: float = 1.51699613e+02
    def k_eff3(self, x):
        """
        Stiffness function obtained from plot_stiffness_function
        
        :param x: Spring extension
        """
        eps = 1e-6
        x *= 1000  # convert to mm
        denom = np.maximum((self.xmax_3 - x), eps)
        return self.k1_3 + self.A_3 * self.p_3 / denom ** (self.p_3 + 1) * 1000

    # action limits 
    tau_max1: float = 0.0286   # N*m (max torque magnitude that can be exerted)
    tau_max2: float = 0.0286
    tau_max3: float = 0.0286

    # def k_eff(self, x):
    #     """
    #     Stiffness function obtained from plot_stiffness_function
        
    #     :param x: Spring extension
    #     """
    #     k0 = 1.07491780e+01
    #     k1 = -1.10957257e+02
    #     A = 9.99276124e-02
    #     p = 1.53491397e+00
    #     xmax = 1.16201538e-01
    #     eps = 1e-6
    #     return k0 + k1*x + A / max((xmax - x), eps)**p
