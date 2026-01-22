import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 0.1155, 1000)
def k_eff1(self, x):
        """
        Stiffness function obtained from plot_stiffness_function

        :param x: Spring extension
        """
        k0 = 1.07491780e+01
        k1 = -1.10957257e+02
        A = 9.99276124e-02
        p = 1.53491397e+00
        xmax = 1.16201538e-01
        eps = 1e-6
        return k0 + k1*x + A / np.maximum((xmax - x), eps)**p

def k_eff2(self, x):
        """
        Stiffness function obtained from plot_stiffness_function

        :param x: Spring extension
        """
        # k0 = 1.07491780e+01
        k0 = 1.37491780e+01
        k1 = -1.10957257e+02
        A = 9.99276124e-02
        p = 1.53491397e+00
        xmax = 1.16201538e-01
        eps = 1e-6
        return k0 + k1*x + A / np.maximum((xmax - x), eps)**p

plt.plot(x, k_eff1(None, x), label='k_eff1')
plt.plot(x, k_eff2(None, x), label='k_eff2')
plt.legend(['k_eff1', 'k_eff2'])
plt.show()