"""
Stiffness fitting using a wall model:

  k(x) = k0 + k1*x + A / max((xmax - x), eps)^p

with eps = 1e-6 for numerical safety near xmax.

Computes tension and ΔL from the cable geometry, derives stiffness (dT/dΔL in N/m),
fits the wall model (uses scipy.optimize.curve_fit if available), and plots data vs fit.
"""

from __future__ import annotations

import math
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None


def l1(theta: float, l: float, l0: float) -> float:
    return math.sqrt(l * l + l0 * l0 - 2.0 * l * l0 * math.cos(theta))


def L(theta: float, l0: float, l: float, r: float, alpha1: float, alpha2: float, beta: float) -> float:
    l1_val = l1(theta, l, l0)
    term0 = math.sqrt(l0 * l0 - r * r)
    term1 = (theta + alpha1 + alpha2 + beta) * r
    term2 = math.sqrt(l1_val * l1_val - r * r)
    return 2.0 * (term0 + term1 + term2)


def T(theta: float, theta_eq: float, K: float, l0: float, alpha2: float, beta: float) -> float:
    denom = 2.0 * l0 * math.sin(alpha2 + beta)
    return K * (theta_eq - theta) / denom


def delta_L(theta: float, L_max: float, l0: float, l: float, r: float, alpha1: float, alpha2: float, beta: float) -> float:
    return L_max - L(theta, l0, l, r, alpha1, alpha2, beta)


def k_model(x: np.ndarray, k0: float, k1: float, A: float, p: float, xmax: float) -> np.ndarray:
    eps = 1e-6
    return k0 + k1 * x + A / np.maximum((xmax - x), eps) ** p


# Constants (mm, N*mm/rad, rad)
L_CONST: Final[dict[str, float | None]] = {
    "l": 74.26,
    "l0": 40.0,
    "r": 3.75,
    "alpha1": math.radians(5.35582),
    "K": 47.12,
    "theta_eq": math.pi / 2,
    "L_max": None,
}

def main() -> None:
    p = L_CONST
    l = float(p["l"])
    l0 = float(p["l0"])
    r = float(p["r"])
    alpha1 = float(p["alpha1"])
    K = float(p["K"])
    theta_eq = float(p["theta_eq"])

    def alpha2(theta_val: float) -> float:
        return math.atan(r / l1(theta_val, l, l0))

    def beta(theta_val: float) -> float:
        l1_val = l1(theta_val, l, l0)
        arg = (-l0 * l0 + l1_val * l1_val + l * l) / (2.0 * l1_val * l)
        arg = max(-1.0, min(1.0, arg))
        return math.acos(arg)

    def L_dyn(theta_val: float) -> float:
        return L(theta_val, l0, l, r, alpha1, alpha2(theta_val), beta(theta_val))

    L_max = float(p["L_max"]) if p["L_max"] is not None else L_dyn(theta_eq)

    thetas = np.linspace(0.0, math.pi / 2, 2000)
    dl_mm = np.array([delta_L(th, L_max, l0, l, r, alpha1, alpha2(th), beta(th)) for th in thetas])
    tensions = np.array([T(th, theta_eq, K, l0, alpha2(th), beta(th)) for th in thetas])

    mask = np.isfinite(dl_mm) & np.isfinite(tensions)
    dl_mm = dl_mm[mask]
    tensions = tensions[mask]

    order = np.argsort(dl_mm)
    dl_mm = dl_mm[order]
    tensions = tensions[order]

    dl_m = dl_mm * 1e-3
    k_vals = np.gradient(tensions, dl_m)  # N/m

    # Data-driven initial guesses
    n = len(dl_m)
    n_lin = max(5, n // 10)
    k0_guess = float(k_vals[0])
    # early slope; guard against zero division
    denom = max(1e-9, dl_m[n_lin - 1] - dl_m[0])
    k1_guess = (k_vals[n_lin - 1] - k_vals[0]) / denom
    span = float(k_vals.max() - k_vals.min())
    A_guess = max(1.0, span)
    p_guess = 1.5
    xmax_guess = float(dl_m.max() * 1.05)

    if curve_fit is not None:
        init = [k0_guess, k1_guess, A_guess, p_guess, xmax_guess]
        try:
            popt, _ = curve_fit(k_model, dl_m, k_vals, p0=init, maxfev=200000)
        except Exception:
            popt = init
    else:
        popt = [k0_guess, k1_guess, A_guess, p_guess, xmax_guess]

    dl_fit_m = np.linspace(dl_m.min(), dl_m.max(), 300)
    k_fit = k_model(dl_fit_m, *popt)
    print("Initial guess [k0, k1, A, p, xmax]:", [k0_guess, k1_guess, A_guess, p_guess, xmax_guess])
    print("Converged params [k0, k1, A, p, xmax]:", popt)

    plt.figure()
    plt.plot(dl_mm, tensions)
    plt.xlabel("ΔL [mm]")
    plt.ylabel("Tension T(θ) [N]")
    plt.title("Tension vs ΔL")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure()
    plt.plot(dl_m, k_vals, label="Stiffness samples")
    plt.plot(dl_fit_m, k_fit, label="Wall model fit")
    plt.xlabel("ΔL [m]")
    plt.ylabel("Stiffness dT/d(ΔL) [N/m]")
    plt.title("Linear Stiffness vs ΔL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
