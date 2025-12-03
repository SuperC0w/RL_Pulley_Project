from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotConfig2D:
    N: int = 800              # keep & draw only the latest N samples
    H_scale: float = 1.4      # axis span as multiple of total link length
    y_limits: tuple = (-2.0, 2.0)
    y_limits_tau: tuple = (-0.15, 0.15)


class MatplotlibRenderer2D:
    """
    Blitting renderer for a 2-link arm. theta1/theta2 are absolute; 0 rad points down.
    """

    def __init__(self, params, cfg: PlotConfig2D | None = None, dt_sim: float = 0.01):
        self.p = params
        self.cfg = cfg or PlotConfig2D()
        self.dt_sim = float(dt_sim)

        self.fig = plt.figure(figsize=(12, 7.5))
        try:
            self.fig.canvas.manager.set_window_title("Pulley Simulator 2D (blitting)")
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            3, 3, width_ratios=[3.0, 2.5, 3.0], wspace=0.35, hspace=0.65
        )
        self.ax_theta1 = self.fig.add_subplot(gs[0, 0])
        self.ax_theta2 = self.fig.add_subplot(gs[1, 0])
        self.ax_phi1 = self.fig.add_subplot(gs[2, 0])
        self.ax_anim = self.fig.add_subplot(gs[:, 1], aspect="equal")
        self.ax_phi2 = self.fig.add_subplot(gs[0, 2])
        self.ax_phi3 = self.fig.add_subplot(gs[1, 2])
        self.ax_tau = self.fig.add_subplot(gs[2, 2])

        win = self.cfg.N * self.dt_sim
        (self.ln_theta1,) = self.ax_theta1.plot([], [], lw=1.0, antialiased=False)
        (self.ln_theta2,) = self.ax_theta2.plot([], [], lw=1.0, antialiased=False)
        (self.ln_phi1,) = self.ax_phi1.plot([], [], lw=1.0, antialiased=False)
        (self.ln_phi2,) = self.ax_phi2.plot([], [], lw=1.0, antialiased=False)
        (self.ln_phi3,) = self.ax_phi3.plot([], [], lw=1.0, antialiased=False)
        (self.ln_tau1,) = self.ax_tau.plot([], [], lw=1.0, antialiased=False, label="tau1")
        (self.ln_tau2,) = self.ax_tau.plot([], [], lw=1.0, antialiased=False, label="tau2")
        (self.ln_tau3,) = self.ax_tau.plot([], [], lw=1.0, antialiased=False, label="tau3")
        self.ax_tau.legend(loc="upper right")

        for ax, lab in (
            (self.ax_theta1, "theta1 (rad)"),
            (self.ax_theta2, "theta2 (rad)"),
            (self.ax_phi1, "phi1 (rad)"),
            (self.ax_phi2, "phi2 (rad)"),
            (self.ax_phi3, "phi3 (rad)"),
            (self.ax_tau, "torques (Nm)"),
        ):
            ax.grid(False)
            ax.set_xlim(-win, 0.0)
            y_lim = self.cfg.y_limits_tau if ax is self.ax_tau else self.cfg.y_limits
            ax.set_ylim(*y_lim)
            ax.set_ylabel(lab)
        self.ax_tau.set_xlabel("time (s)")

        self._build_static_anim()
        self.fig.tight_layout()

        N = self.cfg.N
        self.t_buf = np.zeros(N, float)
        self.th1_buf = np.zeros(N, float)
        self.th2_buf = np.zeros(N, float)
        self.p1_buf = np.zeros(N, float)
        self.p2_buf = np.zeros(N, float)
        self.p3_buf = np.zeros(N, float)
        self.tau1_buf = np.zeros(N, float)
        self.tau2_buf = np.zeros(N, float)
        self.tau3_buf = np.zeros(N, float)
        self.t_idx = 0

        for ln in (
            self.link1_ln,
            self.link2_ln,
            self.joint_mid,
            self.joint_tip,
            self.ln_theta1,
            self.ln_theta2,
            self.ln_phi1,
            self.ln_phi2,
            self.ln_phi3,
            self.ln_tau1,
            self.ln_tau2,
            self.ln_tau3,
        ):
            ln.set_animated(True)

        self._bg_anim = self._bg_theta1 = self._bg_theta2 = self._bg_phi1 = None
        self._bg_phi2 = self._bg_phi3 = self._bg_tau = None
        self._init_blit()
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    def draw_static(self):
        pass

    def refresh_static(self):
        self._build_static_anim()
        self._init_blit()

    def update(self, obs, action=None, goal=None, t: float = 0.0):
        theta1 = float(obs[0])
        theta2 = float(obs[2])
        phi1 = float(obs[4])
        phi2 = float(obs[6])
        phi3 = float(obs[8])

        if action is None:
            action = np.zeros(3, dtype=float)
        action = np.asarray(action, dtype=float).ravel()
        if action.size < 3:
            padded = np.zeros(3, dtype=float)
            padded[: action.size] = action
            action = padded
        tau1, tau2, tau3 = (float(action[0]), float(action[1]), float(action[2]))

        x1, y1, x2, y2 = self._fk(theta1, theta2)
        self.link1_ln.set_data([0.0, x1], [0.0, y1])
        self.link2_ln.set_data([x1, x2], [y1, y2])
        self.joint_mid.set_data([x1], [y1])
        self.joint_tip.set_data([x2], [y2])

        last_i = self._rb_append(t, theta1, theta2, phi1, phi2, phi3, tau1, tau2, tau3)
        T, Th1, Th2, P1, P2, P3, Tau1, Tau2, Tau3 = self._rb_latestN(last_i)

        if T.size:
            X = T - T[-1]
            self.ln_theta1.set_data(X, Th1)
            self.ln_theta2.set_data(X, Th2)
            self.ln_phi1.set_data(X, P1)
            self.ln_phi2.set_data(X, P2)
            self.ln_phi3.set_data(X, P3)
            self.ln_tau1.set_data(X, Tau1)
            self.ln_tau2.set_data(X, Tau2)
            self.ln_tau3.set_data(X, Tau3)
        else:
            for ln in (
                self.ln_theta1,
                self.ln_theta2,
                self.ln_phi1,
                self.ln_phi2,
                self.ln_phi3,
                self.ln_tau1,
                self.ln_tau2,
                self.ln_tau3,
            ):
                ln.set_data([], [])

        c = self.fig.canvas
        c.restore_region(self._bg_anim)
        self.ax_anim.draw_artist(self.link1_ln)
        self.ax_anim.draw_artist(self.link2_ln)
        self.ax_anim.draw_artist(self.joint_mid)
        self.ax_anim.draw_artist(self.joint_tip)
        c.blit(self.ax_anim.bbox)

        c.restore_region(self._bg_theta1)
        self.ax_theta1.draw_artist(self.ln_theta1)
        c.blit(self.ax_theta1.bbox)

        c.restore_region(self._bg_theta2)
        self.ax_theta2.draw_artist(self.ln_theta2)
        c.blit(self.ax_theta2.bbox)

        c.restore_region(self._bg_phi1)
        self.ax_phi1.draw_artist(self.ln_phi1)
        c.blit(self.ax_phi1.bbox)

        c.restore_region(self._bg_phi2)
        self.ax_phi2.draw_artist(self.ln_phi2)
        c.blit(self.ax_phi2.bbox)

        c.restore_region(self._bg_phi3)
        self.ax_phi3.draw_artist(self.ln_phi3)
        c.blit(self.ax_phi3.bbox)

        c.restore_region(self._bg_tau)
        self.ax_tau.draw_artist(self.ln_tau1)
        self.ax_tau.draw_artist(self.ln_tau2)
        self.ax_tau.draw_artist(self.ln_tau3)
        c.blit(self.ax_tau.bbox)
        c.flush_events()

    def close(self):
        plt.close(self.fig)

    def _build_static_anim(self):
        l1 = float(self.p.l1)
        l2 = float(self.p.l2)
        span = (l1 + l2) * self.cfg.H_scale

        self.ax_anim.cla()
        self.ax_anim.set_title("Two-link arm")
        self.ax_anim.set_xlim(-span, span)
        self.ax_anim.set_ylim(-span, span)
        self.ax_anim.set_aspect("equal")

        (self.link1_ln,) = self.ax_anim.plot([0.0, 0.0], [0.0, -l1], lw=2.0, zorder=3, antialiased=True)
        (self.link2_ln,) = self.ax_anim.plot([0.0, 0.0], [-l1, -(l1 + l2)], lw=2.0, zorder=3, antialiased=True)
        (self.joint_mid,) = self.ax_anim.plot([0.0], [-l1], marker="o", ms=4, color="black", zorder=5)
        (self.joint_tip,) = self.ax_anim.plot([0.0], [-(l1 + l2)], marker="o", ms=4, color="black", zorder=5)
        self.ax_anim.plot(0.0, 0.0, marker="o", ms=4, color="black", zorder=6)

    def _init_blit(self):
        self.fig.canvas.draw()
        c = self.fig.canvas
        self._bg_anim = c.copy_from_bbox(self.ax_anim.bbox)
        self._bg_theta1 = c.copy_from_bbox(self.ax_theta1.bbox)
        self._bg_theta2 = c.copy_from_bbox(self.ax_theta2.bbox)
        self._bg_phi1 = c.copy_from_bbox(self.ax_phi1.bbox)
        self._bg_phi2 = c.copy_from_bbox(self.ax_phi2.bbox)
        self._bg_phi3 = c.copy_from_bbox(self.ax_phi3.bbox)
        self._bg_tau = c.copy_from_bbox(self.ax_tau.bbox)

    def _on_resize(self, *_):
        self._init_blit()

    def _rb_append(self, t, th1, th2, p1, p2, p3, tau1, tau2, tau3):
        i = self.t_idx % self.cfg.N
        self.t_buf[i] = t
        self.th1_buf[i] = th1
        self.th2_buf[i] = th2
        self.p1_buf[i] = p1
        self.p2_buf[i] = p2
        self.p3_buf[i] = p3
        self.tau1_buf[i] = tau1
        self.tau2_buf[i] = tau2
        self.tau3_buf[i] = tau3
        self.t_idx += 1
        return i

    def _rb_latestN(self, last_i):
        N = self.cfg.N
        if self.t_idx < N:
            T = self.t_buf[: self.t_idx]
            Th1 = self.th1_buf[: self.t_idx]
            Th2 = self.th2_buf[: self.t_idx]
            P1 = self.p1_buf[: self.t_idx]
            P2 = self.p2_buf[: self.t_idx]
            P3 = self.p3_buf[: self.t_idx]
            Tau1 = self.tau1_buf[: self.t_idx]
            Tau2 = self.tau2_buf[: self.t_idx]
            Tau3 = self.tau3_buf[: self.t_idx]
        else:
            if last_i + 1 < N:
                sl = slice(last_i + 1, None)
                T = np.concatenate((self.t_buf[sl], self.t_buf[: last_i + 1]))
                Th1 = np.concatenate((self.th1_buf[sl], self.th1_buf[: last_i + 1]))
                Th2 = np.concatenate((self.th2_buf[sl], self.th2_buf[: last_i + 1]))
                P1 = np.concatenate((self.p1_buf[sl], self.p1_buf[: last_i + 1]))
                P2 = np.concatenate((self.p2_buf[sl], self.p2_buf[: last_i + 1]))
                P3 = np.concatenate((self.p3_buf[sl], self.p3_buf[: last_i + 1]))
                Tau1 = np.concatenate((self.tau1_buf[sl], self.tau1_buf[: last_i + 1]))
                Tau2 = np.concatenate((self.tau2_buf[sl], self.tau2_buf[: last_i + 1]))
                Tau3 = np.concatenate((self.tau3_buf[sl], self.tau3_buf[: last_i + 1]))
            else:
                T = self.t_buf
                Th1 = self.th1_buf
                Th2 = self.th2_buf
                P1 = self.p1_buf
                P2 = self.p2_buf
                P3 = self.p3_buf
                Tau1 = self.tau1_buf
                Tau2 = self.tau2_buf
                Tau3 = self.tau3_buf
        return T, Th1, Th2, P1, P2, P3, Tau1, Tau2, Tau3

    def _fk(self, theta1, theta2):
        l1 = float(self.p.l1)
        l2 = float(self.p.l2)
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        return x1, y1, x2, y2

    def clear_buffers(self):
        self.t_idx = 0
        self.t_buf.fill(0.0)
        self.th1_buf.fill(0.0)
        self.th2_buf.fill(0.0)
        self.p1_buf.fill(0.0)
        self.p2_buf.fill(0.0)
        self.p3_buf.fill(0.0)
        self.tau1_buf.fill(0.0)
        self.tau2_buf.fill(0.0)
        self.tau3_buf.fill(0.0)

        self.link1_ln.set_data([0.0, 0.0], [0.0, -float(self.p.l1)])
        self.link2_ln.set_data([0.0, 0.0], [-float(self.p.l1), -(float(self.p.l1) + float(self.p.l2))])
        self.joint_mid.set_data([0.0], [-float(self.p.l1)])
        self.joint_tip.set_data([0.0], [-(float(self.p.l1) + float(self.p.l2))])

        for ln in (
            self.ln_theta1,
            self.ln_theta2,
            self.ln_phi1,
            self.ln_phi2,
            self.ln_phi3,
            self.ln_tau1,
            self.ln_tau2,
            self.ln_tau3,
        ):
            ln.set_data([], [])

        self._init_blit()
        c = self.fig.canvas
        c.restore_region(self._bg_anim)
        self.ax_anim.draw_artist(self.link1_ln)
        self.ax_anim.draw_artist(self.link2_ln)
        self.ax_anim.draw_artist(self.joint_mid)
        self.ax_anim.draw_artist(self.joint_tip)
        c.blit(self.ax_anim.bbox)
        for bg, ax, ln in (
            (self._bg_theta1, self.ax_theta1, self.ln_theta1),
            (self._bg_theta2, self.ax_theta2, self.ln_theta2),
            (self._bg_phi1, self.ax_phi1, self.ln_phi1),
            (self._bg_phi2, self.ax_phi2, self.ln_phi2),
            (self._bg_phi3, self.ax_phi3, self.ln_phi3),
            (self._bg_tau, self.ax_tau, self.ln_tau1),
        ):
            c.restore_region(bg)
            ax.draw_artist(ln)
            c.blit(ax.bbox)
        c.flush_events()
