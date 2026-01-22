import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None

from tqdm import tqdm

from env2d.dynamics import make_dynamics, Inputs
from env2d.integrators import rk4_step
from env2d.params import PulleyParams


def pwm_to_tau(pwm: np.ndarray) -> np.ndarray:
    return pwm / 1750.0 * 0.0286


def load_log(path: Path, dt: float, stride: int = 1, max_rows: int | None = None):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if max_rows is not None:
        data = data[:max_rows]
    if stride > 1:
        data = data[::stride]

    tau1 = pwm_to_tau(data["m1_pwm"])
    tau2 = pwm_to_tau(data["m2_pwm"])
    tau3 = pwm_to_tau(data["m3_pwm"])
    torques = np.stack([tau1, tau2, tau3], axis=1)

    theta1 = np.deg2rad(data["enc1_deg"])
    theta2 = np.deg2rad(data["enc2_deg"])

    t = np.arange(theta1.size, dtype=float) * dt * stride
    return t, torques, theta1, theta2


def initial_state_from_theta(theta1, theta2, params: PulleyParams, dt: float, theta1_next=None, theta2_next=None):
    x = np.zeros(10, dtype=float)
    x[0] = theta1
    x[2] = theta2
    if theta1_next is not None:
        x[1] = (theta1_next - theta1) / dt
    if theta2_next is not None:
        x[3] = (theta2_next - theta2) / dt

    # Wind motor pulleys to be consistent with theta1/theta2 at t0.
    x[4] = -theta1 * params.r4 / params.r1 - theta2 * params.r5 / params.r1
    x[6] = -theta1 * params.r4 / params.r2 + theta2 * params.r5 / params.r2
    x[8] = theta1 * params.r4 / params.r3
    return x


class ParamWrapper:
    def __init__(
        self,
        base: PulleyParams,
        k1_1,
        A_1,
        p_1,
        xmax_1,
        k1_2,
        A_2,
        p_2,
        xmax_2,
        k1_3,
        A_3,
        p_3,
        xmax_3,
        I4_scale,
        I5_scale,
        I1_scale,
        I2_scale,
        I3_scale,
        m1_scale,
        m2_scale,
        m3_scale,
        m4_scale,
        m5_scale,
        F1,
        c1,
        c2,
        c3,
        c4,
        c5,
    ):
        self._base = base
        self._k1_1 = float(k1_1)
        self._A_1 = float(A_1)
        self._p_1 = float(p_1)
        self._xmax_1 = float(xmax_1)
        self._k1_2 = float(k1_2)
        self._A_2 = float(A_2)
        self._p_2 = float(p_2)
        self._xmax_2 = float(xmax_2)
        self._k1_3 = float(k1_3)
        self._A_3 = float(A_3)
        self._p_3 = float(p_3)
        self._xmax_3 = float(xmax_3)
        self._I4_scale = float(I4_scale)
        self._I5_scale = float(I5_scale)
        self._I1_scale = float(I1_scale)
        self._I2_scale = float(I2_scale)
        self._I3_scale = float(I3_scale)
        self._m1_scale = float(m1_scale)
        self._m2_scale = float(m2_scale)
        self._m3_scale = float(m3_scale)
        self._m4_scale = float(m4_scale)
        self._m5_scale = float(m5_scale)
        self.F1 = float(F1)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.c4 = float(c4)
        self.c5 = float(c5)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _k_eff(self, x, k1, A, p, xmax):
        eps = 1e-6
        x_mm = x * 1000.0
        denom = np.maximum((xmax - x_mm), eps)
        return k1 + A * p / denom ** (p + 1.0) * 1000.0

    def k_eff1(self, x):
        return self._k_eff(x, self._k1_1, self._A_1, self._p_1, self._xmax_1)

    def k_eff2(self, x):
        return self._k_eff(x, self._k1_2, self._A_2, self._p_2, self._xmax_2)

    def k_eff3(self, x):
        return self._k_eff(x, self._k1_3, self._A_3, self._p_3, self._xmax_3)

    def get_I4(self, theta):
        r4 = self._base.r4
        r5 = self._base.r5
        l1 = self._base.l1
        l2 = self._base.l2
        w1 = self._base.w1
        w2 = self._base.w2

        m4 = self._base.m4 * self._m4_scale
        m5 = self._base.m5 * self._m5_scale
        m_link1 = self._base.m_link1
        m_link2 = self._base.m_link2

        I4 = (
            m4 * r4**2 / 2.0
            + (m_link1 * (l1**2 + w1**2) / 12.0 + m_link1 * (l1 / 2.0) ** 2)
            + m5 * r5**2 / 2.0
            + m5 * l1**2
            + m_link2
            * (
                (l2**2 + w2**2) / 12.0
                + l1**2
                + (l2 / 2.0) ** 2
                + 2.0 * l1 * l2 / 2.0 * np.cos(theta)
            )
        )
        return self._I4_scale * I4

    @property
    def I5(self):
        r5 = self._base.r5
        l2 = self._base.l2
        w2 = self._base.w2
        m5 = self._base.m5 * self._m5_scale
        m_link2 = self._base.m_link2
        I5 = (m5 * r5**2 / 2.0 + (m_link2 * (l2**2 + w2**2) / 12.0 + m_link2 * (l2 / 2.0) ** 2)) * 2.0
        return self._I5_scale * I5

    @property
    def I1(self):
        r1 = self._base.r1
        m1 = self._base.m1 * self._m1_scale
        motor_I = 11 / 10000000
        return self._I1_scale * (motor_I + m1 * r1**2 / 2.0)

    @property
    def I2(self):
        r2 = self._base.r2
        m2 = self._base.m2 * self._m2_scale
        motor_I = 11 / 10000000
        return self._I2_scale * (motor_I + m2 * r2**2 / 2.0)

    @property
    def I3(self):
        r3 = self._base.r3
        m3 = self._base.m3 * self._m3_scale
        motor_I = 11 / 10000000
        return self._I3_scale * (motor_I + m3 * r3**2 / 2.0)


def simulate(params: ParamWrapper, torques: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, dt: float):
    inputs = Inputs()
    f = make_dynamics(params, inputs)

    theta1_next = theta1[1] if theta1.size > 1 else None
    theta2_next = theta2[1] if theta2.size > 1 else None
    x = initial_state_from_theta(theta1[0], theta2[0], params, dt, theta1_next, theta2_next)

    th1_pred = np.zeros(theta1.size, dtype=float)
    th2_pred = np.zeros(theta2.size, dtype=float)
    th1_pred[0] = x[0]
    th2_pred[0] = x[2]

    for i in range(1, theta1.size):
        tau = torques[i]
        inputs.tau1 = float(np.clip(tau[0], -params.tau_max1, params.tau_max1))
        inputs.tau2 = float(np.clip(tau[1], -params.tau_max2, params.tau_max2))
        inputs.tau3 = float(np.clip(tau[2], -params.tau_max3, params.tau_max3))
        x = rk4_step(f, x, dt)
        th1_pred[i] = x[0]
        th2_pred[i] = x[2]

        if not np.isfinite(th1_pred[i]) or not np.isfinite(th2_pred[i]):
            return None

    return th1_pred, th2_pred


def build_objective(logs, dt):
    base = PulleyParams()

    def residuals(v):
        (
            k1_1,
            A_1,
            p_1,
            xmax_1,
            k1_2,
            A_2,
            p_2,
            xmax_2,
            k1_3,
            A_3,
            p_3,
            xmax_3,
            I4_scale,
            I5_scale,
            I1_scale,
            I2_scale,
            I3_scale,
            m1_scale,
            m2_scale,
            m3_scale,
            m4_scale,
            m5_scale,
            F1,
            c1,
            c2,
            c3,
            c4,
            c5,
        ) = v
        if (
            A_1 <= 0
            or p_1 <= 0
            or xmax_1 <= 0
            or A_2 <= 0
            or p_2 <= 0
            or xmax_2 <= 0
            or A_3 <= 0
            or p_3 <= 0
            or xmax_3 <= 0
            or I4_scale <= 0
            or I5_scale <= 0
            or I1_scale <= 0
            or I2_scale <= 0
            or I3_scale <= 0
            or m1_scale <= 0
            or m2_scale <= 0
            or m3_scale <= 0
            or m4_scale <= 0
            or m5_scale <= 0
            or not np.isfinite(F1)
            or c1 < 0
            or c2 < 0
            or c3 < 0
            or c4 < 0
            or c5 < 0
        ):
            return np.full(10, 1e6, dtype=float)

        params = ParamWrapper(
            base,
            k1_1,
            A_1,
            p_1,
            xmax_1,
            k1_2,
            A_2,
            p_2,
            xmax_2,
            k1_3,
            A_3,
            p_3,
            xmax_3,
            I4_scale,
            I5_scale,
            I1_scale,
            I2_scale,
            I3_scale,
            m1_scale,
            m2_scale,
            m3_scale,
            m4_scale,
            m5_scale,
            F1,
            c1,
            c2,
            c3,
            c4,
            c5,
        )
        res_list = []
        for _, torques, th1, th2 in logs:
            sim = simulate(params, torques, th1, th2, dt)
            if sim is None:
                return np.full(10, 1e6, dtype=float)
            th1_pred, th2_pred = sim
            res_list.append(th1_pred - th1)
            res_list.append(th2_pred - th2)
        return np.concatenate(res_list)

    return residuals


def main():
    parser = argparse.ArgumentParser(description="Fit stiffness and inertia parameters for env2d.")
    parser.add_argument("--logs", nargs="*", default=None, help="CSV log paths.")
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep (s).")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth sample.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows per log.")
    parser.add_argument("--method", choices=["least_squares", "random"], default="least_squares")
    parser.add_argument("--random-iters", type=int, default=500, help="Random search iterations.")
    parser.add_argument("--max-nfev", type=int, default=200, help="Max function evals for least squares.")
    parser.add_argument("--progress", action="store_true", help="Show a tqdm progress bar.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    parser.add_argument("--out", default="fit_params.json", help="Output JSON path.")
    args = parser.parse_args()

    if args.logs is None or len(args.logs) == 0:
        args.logs = sorted(str(p) for p in (SCRIPT_DIR / "data").glob("serial_log_*.csv"))
    if len(args.logs) == 0:
        raise ValueError("No logs found. Provide --logs or place CSVs under real2sim/data.")

    logs = []
    for p in args.logs:
        path = Path(p)
        t, torques, th1, th2 = load_log(path, args.dt, args.stride, args.max_rows)
        logs.append((t, torques, th1, th2))

    base = PulleyParams()
    init = np.array([
        base.k1_1,
        base.A_1,
        base.p_1,
        base.xmax_1,
        base.k1_2,
        base.A_2,
        base.p_2,
        base.xmax_2,
        base.k1_3,
        base.A_3,
        base.p_3,
        base.xmax_3,
        1.0,  # I4_scale
        1.0,  # I5_scale
        1.0,  # I1_scale
        1.0,  # I2_scale
        1.0,  # I3_scale
        1.0,  # m1_scale
        1.0,  # m2_scale
        1.0,  # m3_scale
        1.0,  # m4_scale
        1.0,  # m5_scale
        base.F1,
        base.c1,
        base.c2,
        base.c3,
        base.c4,
        base.c5,
    ], dtype=float)

    bounds = (
        np.array([
            0.0, 1e-8, 0.2, 80.0,   # spring 1
            0.0, 1e-8, 0.2, 80.0,   # spring 2
            0.0, 1e-8, 0.2, 80.0,   # spring 3
            0.2, 0.2,              # I4/I5 scale
            0.2, 0.2, 0.2,         # I1/I2/I3 scale
            0.2, 0.2, 0.2, 0.2, 0.2,  # m1..m5 scale
            -5.0,  # F1
            0.0, 0.0, 0.0, 0.0, 0.0,  # c1..c5
        ], dtype=float),
        np.array([
            1.0, 1e3, 5.0, 500.0,  # spring 1
            1.0, 1e3, 5.0, 500.0,  # spring 2
            1.0, 1e3, 5.0, 500.0,  # spring 3
            5.0, 5.0,              # I4/I5 scale
            5.0, 5.0, 5.0,         # I1/I2/I3 scale
            5.0, 5.0, 5.0, 5.0, 5.0,  # m1..m5 scale
            5.0,   # F1
            1e-2, 1e-2, 1e-2, 1e-3, 1e-3,  # c1..c5
        ], dtype=float),
    )

    objective = build_objective(logs, args.dt * args.stride)

    progress = None
    if args.method == "least_squares":
        total = args.max_nfev * (init.size + 1)
    else:
        total = args.random_iters
    if not args.no_progress:
        progress = tqdm(total=total, desc="fit_params", unit="eval")

    def objective_with_progress(v):
        if progress is not None:
            progress.update(1)
        return objective(v)

    if args.method == "least_squares" and least_squares is not None:
        result = least_squares(objective_with_progress, init, bounds=bounds, max_nfev=args.max_nfev)
        best = result.x
        err = np.mean(objective(best) ** 2)
    else:
        rng = np.random.default_rng(0)
        low, high = bounds
        best = init.copy()
        best_err = np.mean(objective(best) ** 2)
        for _ in range(args.random_iters):
            if progress is not None:
                progress.update(1)
            cand = low + rng.random(low.size) * (high - low)
            err = np.mean(objective(cand) ** 2)
            if err < best_err:
                best_err = err
                best = cand
        err = best_err
    if progress is not None:
        progress.close()

    out = {
        "k1_1": float(best[0]),
        "A_1": float(best[1]),
        "p_1": float(best[2]),
        "xmax_1": float(best[3]),
        "k1_2": float(best[4]),
        "A_2": float(best[5]),
        "p_2": float(best[6]),
        "xmax_2": float(best[7]),
        "k1_3": float(best[8]),
        "A_3": float(best[9]),
        "p_3": float(best[10]),
        "xmax_3": float(best[11]),
        "I4_scale": float(best[12]),
        "I5_scale": float(best[13]),
        "I1_scale": float(best[14]),
        "I2_scale": float(best[15]),
        "I3_scale": float(best[16]),
        "m1_scale": float(best[17]),
        "m2_scale": float(best[18]),
        "m3_scale": float(best[19]),
        "m4_scale": float(best[20]),
        "m5_scale": float(best[21]),
        "F1": float(best[22]),
        "c1": float(best[23]),
        "c2": float(best[24]),
        "c3": float(best[25]),
        "c4": float(best[26]),
        "c5": float(best[27]),
        "mse": float(err),
        "logs": args.logs,
        "dt": args.dt,
        "stride": args.stride,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
