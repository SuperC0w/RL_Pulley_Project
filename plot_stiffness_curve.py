import numpy as np
import matplotlib.pyplot as plt

# local helper so it’s available to the ODE function
def k_eff(k0, x, alpha):
    # return k0 * (1.0 + alpha * x * x)
    return k0 * (alpha * x * x)
def k_eff2(k0, x, alpha):
    return k0 * (1.0 + alpha * x * x)
    # return k0 * (alpha * x * x)

if __name__ == "__main__":
    alpha = 5e4
    k = 1000
    N = 100
    end = 0.0036

    x = np.linspace(0, end, N)
    k = k_eff(k, x, alpha)
    F = x*k
    plt.plot(x,k)

    plt.figure()
    plt.title("Force-Displacement Curve")
    plt.plot(x, F)
    
    alpha = 8e5
    k = 100
    N = 100
    end = 0.0036

    x = np.linspace(0, end, N)
    k = k_eff2(k, x, alpha)
    F = x*k
    plt.figure()
    plt.plot(x,k)

    plt.figure()
    plt.title("Force-Displacement Curve")
    plt.plot(x, F)

    
    
    plt.show()


# Note->Current model trained on
# stiffness function: k0 * (1.0 + alpha * x * x)
# alpha = 1e4
# k0 = 1000

# Another potential combination
# stiffness function: k0 * (alpha * x * x)
# alpha = 5e4
# k0 = 1000

# Previously tested potential combination
# stiffness function: k0 * (1 + alpha * x * x)
# alpha = 5e5
# k0 = 100

# Currently tested potential combination
# stiffness function: k0 * (1 + alpha * x * x)
# alpha = 8e5
# k0 = 100