import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

# --- Academic Plot Formatting ---
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 14})

# --- Theoretical Curve Calculation ---
sample_num = 1000
l = 148.52273 / 2
l0 = 40
r = 3.75
K = 1.85 * l0 / (np.pi / 2)
theta_eq = np.pi / 2

theta = np.linspace(-np.arcsin(r / l0), theta_eq, sample_num)
theta = theta[10:]

l1 = np.sqrt(l**2 + l0**2 - 2 * l * l0 * np.cos(theta))
alpha1 = np.arcsin(r / l0)
alpha2 = np.arcsin(r / l1)
beta = np.arcsin(l0 * np.sin(theta) / l1)
L_theory = 2 * (np.sqrt(l0**2 - r**2) + (theta + alpha1 + alpha2 + beta) * r + np.sqrt(l1**2 - r**2))
T_theory = K * (theta_eq - theta) / (2 * l * np.sin(alpha2 + beta))
deltaL_theory = L_theory[-1] - L_theory

# --- Experimental Data Loading and Processing ---
# Load CSV files
vsm1_df = pd.read_csv('motor1.csv')
vsm2_df = pd.read_csv('motor2.csv')
vsm3_df = pd.read_csv('motor3.csv')

vsm1_df['Tension'] = vsm1_df['pwm_value'] * 7.5 / 1600
vsm2_df['Tension'] = vsm2_df['pwm_value'] * 7.5 / 1600
vsm3_df['Tension'] = vsm3_df['pwm_value'] * 7.5 / 1600

CONVERSION_FACTOR = 6 * 2 * np.pi / 360
vsm1_df['deltaL'] = vsm1_df['motor_position'] * CONVERSION_FACTOR
vsm2_df['deltaL'] = vsm2_df['motor_position'] * CONVERSION_FACTOR
vsm3_df['deltaL'] = vsm3_df['motor_position'] * CONVERSION_FACTOR

vsm1_df = vsm1_df.sort_values(by='deltaL')
vsm2_df = vsm2_df.sort_values(by='deltaL')
vsm3_df = vsm3_df.sort_values(by='deltaL')

# --- Curve fitting (wall model) ---
def wall_model(x, k0, k1, A, p, xmax):
    """
    Wall model function used to generate function for tension over displacement.
    """
    eps = 1e-6
    return k0 + k1 * x + A / np.maximum((xmax - x), eps) ** p

def wall_model_stiffness(x, k0, k1, A, p, xmax):
    """
    Derivative of wall model function used to generate stiffness over displacement."""
    eps = 1e-6
    denom = np.maximum((xmax - x), eps)
    return k1 + A * p / denom ** (p + 1)

def _initial_guess(x, y):
    n = len(x)
    n_lin = max(5, n // 10)
    k0_guess = float(y[0])
    denom = max(1e-9, x[n_lin - 1] - x[0])
    k1_guess = float((y[n_lin - 1] - y[0]) / denom)
    span = float(y.max() - y.min())
    A_guess = max(1e-6, span)
    p_guess = 1.5
    xmax_guess = float(x.max() * 1.05)
    return [k0_guess, k1_guess, A_guess, p_guess, xmax_guess]

def fit_and_plot_wall(df, color, label):
    x = df['deltaL'].to_numpy()
    y = df['Tension'].to_numpy()
    if x.size < 5:
        return None
    init = _initial_guess(x, y)
    if curve_fit is not None:
        try:
            popt, _ = curve_fit(wall_model, x, y, p0=init, maxfev=200000)
        except Exception:
            popt = init
    else:
        popt = init
    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = wall_model(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=color, label=f'{label} (Wall fit)')
    return popt

# --- Plotting ---
plt.figure()

# Plot theoretical curve
plt.plot(deltaL_theory, T_theory, 'k-', linewidth=3, label='Theoretical Model')

# Plot experimental data
plt.plot(vsm1_df['deltaL'], vsm1_df['Tension'], 'r--', linewidth=1.5, label='VSM 1 (Exp.)')
plt.plot(vsm2_df['deltaL'], vsm2_df['Tension'], 'g:', linewidth=1.5, label='VSM 2 (Exp.)')
plt.plot(vsm3_df['deltaL'], vsm3_df['Tension'], 'b-.', linewidth=1.5, label='VSM 3 (Exp.)')

# Plot fitted curves
fit1 = fit_and_plot_wall(vsm1_df, 'r', 'VSM 1')
fit2 = fit_and_plot_wall(vsm2_df, 'g', 'VSM 2')
fit3 = fit_and_plot_wall(vsm3_df, 'b', 'VSM 3')

# Formatting
plt.xlabel(r'Cable Displacement $\Delta L$ (mm)')
plt.ylabel(r'Cable Tension $T$ (N)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(0, 160)
plt.ylim(0, 3)

# --- Plot stiffness from fitted wall models ---
def plot_stiffness(df, fit_params, color, label):
    if fit_params is None:
        return
    x = df['deltaL'].to_numpy()
    print(label)
    print(fit_params)
    x_fit = np.linspace(x.min(), x.max(), 400)
    k_fit = wall_model_stiffness(x_fit, *fit_params) * 1000.0
    plt.plot(x_fit, k_fit, color=color, linewidth=2.0, label=f'{label} (Wall dT/dΔ)')

plt.figure()
plot_stiffness(vsm1_df, fit1, 'r', 'VSM 1')
plot_stiffness(vsm2_df, fit2, 'g', 'VSM 2')
plot_stiffness(vsm3_df, fit3, 'b', 'VSM 3')
plt.xlabel(r'Cable Displacement $\Delta L$ (mm)')
plt.ylabel(r'Stiffness $dT/d\Delta L$ (N/m)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('VSM_Characteristic_Curve_Comparison.pdf', bbox_inches='tight')
plt.show()
