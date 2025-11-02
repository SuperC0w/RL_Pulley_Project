import numpy as np
import matplotlib.pyplot as plt

vel_limit = np.deg2rad(720.0)
vel_soft_margin_ratio = 0.5               
w_vel_safety = 20.0  

dtheta = np.linspace(0, 5*vel_limit, 1000)

limit = float(vel_limit)
margin = float(vel_soft_margin_ratio) * limit

y = w_vel_safety * (((dtheta - margin) / (limit - margin)) ** 2)

plt.plot(dtheta, y)

plt.show()