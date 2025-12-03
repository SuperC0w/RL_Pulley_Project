import numpy as np

def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def semi_implicit_euler_step(f, x, dt):
    # assumes state is [q, dq, ...] pairs; for generality, just do explicit Euler
    return x + dt * f(x)
