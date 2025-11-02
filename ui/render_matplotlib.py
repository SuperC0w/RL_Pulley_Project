from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

@dataclass
class PlotConfig:
    N: int = 800              # keep & draw only the latest N samples
    H_scale: float = 1.4      # column height as multiple of link length
    y_limits: tuple = (-2.0, 2.0)  # fixed y-range for time plots
    y_limits_coact: tuple = (0, 10)
    y_limits_tau: tuple = (-0.01, 0.130)

    # Spring appearance
    spring_coils: int = 8         # number of zig-zag “peaks”
    spring_amp: float = 0.005      # amplitude in world units
    spring_lw: float = 1.0        # spring linewidth
    spring_start: float = 0.04   # spring starting location
    spring_length: float = 0.05   # spring length

class MatplotlibRenderer:
    """
    Blitting-only renderer.
    Assumes obs = [theta, dtheta, phi1, dphi1, phi2, dphi2].
    Layout:
      - Left column: θ(t), φ1(t), φ2(t) (fixed x in seconds: [-N*dt, 0])
      - Middle: big pulley + two small pulleys + link (θ=0 points down)
      - Right: blank (reserved for future in-figure controls)
    """

    def __init__(self, params, cfg: PlotConfig | None = None, dt_sim: float = 0.01):
        self.p = params
        self.cfg = cfg or PlotConfig()
        self.dt_sim = float(dt_sim)

        # ---- Figure / axes ----
        self.fig = plt.figure(figsize=(12, 7.5))
        try:
            self.fig.canvas.manager.set_window_title("Pulley Simulator (blitting)")
        except Exception:
            pass
        gs = self.fig.add_gridspec(
            3, 3, width_ratios=[3.0, 2.5, 3.0], wspace=0.35, hspace=0.65
        )
        self.ax_theta = self.fig.add_subplot(gs[0, 0])
        self.ax_phi1  = self.fig.add_subplot(gs[1, 0])
        self.ax_phi2  = self.fig.add_subplot(gs[2, 0])
        self.ax_anim  = self.fig.add_subplot(gs[:, 1], aspect="equal")
        self.ax_coact = self.fig.add_subplot(gs[0, 2])
        self.ax_tau1 = self.fig.add_subplot(gs[1, 2])
        self.ax_tau2 = self.fig.add_subplot(gs[2, 2])

        # ---- Time plots (fixed axes; no autoscale) ----
        win = self.cfg.N * self.dt_sim
        self.ln_theta, self.ln_theta_goal = self.ax_theta.plot([], [],
                                              [], [],
                                                lw=1.0, antialiased=False)
        self.ln_theta_goal.set_linestyle((0, (12, 10)))
        (self.ln_phi1,)  = self.ax_phi1.plot([], [], lw=1.0, antialiased=False)
        (self.ln_phi2,)  = self.ax_phi2.plot([], [], lw=1.0, antialiased=False)
        self.ln_coact, self.ln_coact_goal = self.ax_coact.plot([], [],
                                               [], [],
                                                lw=1.0, antialiased=False)
        self.ln_coact_goal.set_linestyle((0, (12, 10)))
        (self.ln_tau1,)  = self.ax_tau1.plot([], [], lw=1.0, antialiased=False)
        (self.ln_tau2,)  = self.ax_tau2.plot([], [], lw=1.0, antialiased=False)
        for ax, lab in ((self.ax_theta, "θ (rad)"),
                        (self.ax_phi1,  "φ₁ (rad)"),
                        (self.ax_phi2,  "φ₂ (rad)"),
                        (self.ax_coact, "Co-activation (N)"),
                        (self.ax_tau1, "Tau1 (Nm)"),
                        (self.ax_tau2, "Tau2 (Nm)")):
            ax.grid(False)
            ax.set_xlim(-win, 0.0)                 # fixed; show last N samples in seconds
            if ax is self.ax_coact:
                y_lim = self.cfg.y_limits_coact
            elif ax is self.ax_tau1 or ax is self.ax_tau2:
                y_lim = self.cfg.y_limits_tau
            else:
                y_lim = self.cfg.y_limits
            ax.set_ylim(*y_lim)        # fixed for speed
            ax.set_ylabel(lab)
        self.ax_phi2.set_xlabel("time (s)")

        # ---- Animation scaffold (static parts) ----
        self._build_static_anim()

        self.fig.tight_layout()

        # ---- Ring buffers (fixed size N) ----
        N = self.cfg.N
        self.t_buf  = np.zeros(N, float); self.t_idx = 0
        self.th_buf = np.zeros(N, float)
        self.th_goal_buf = np.zeros(N, float)
        self.p1_buf = np.zeros(N, float)
        self.p2_buf = np.zeros(N, float)
        self.coact_buf = np.zeros(N, float)
        self.coact_goal_buf = np.zeros(N, float)
        self.tau1_buf = np.zeros(N, float)
        self.tau2_buf = np.zeros(N, float)

        # ---- Blitting setup ----
        for ln in (self.link_ln, self.ln_theta, self.ln_theta_goal, self.ln_phi1, self.ln_phi2, self.ln_coact, self.ln_coact_goal, 
                   self.ln_tau1, self.ln_tau2):
            ln.set_animated(True)

        self._bg_anim = self._bg_theta = self._bg_phi1 = self._bg_phi2 = self._bg_coact = self._bg_tau1 = self._bg_tau2 = None
        self._init_blit()
        # Re-capture backgrounds if the window is resized
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    # ---------- public API ----------
    def draw_static(self):
        # Nothing else to do; static elements already added and backgrounds captured.
        pass

    def refresh_static(self):
        """
        Call this once after changing static geometry params (e.g., r1/r2/r3/l)
        to update patches and re-capture blit backgrounds.
        """
        # Update static geometry from current params
        R3 = float(self.p.r3)
        L  = float(self.p.l)
        r1 = float(self.p.r1); r2 = float(self.p.r2)
        H  = L * self.cfg.H_scale

        # Update circle radii
        self.wheel_big.set_radius(R3)
        self.wheel_p1.set_radius(r1)
        self.wheel_p2.set_radius(r2)

        # Update column tops (re-draw lines by setting data)
        self.col_left.set_data([-R3, -R3], [0.0, H - r1])
        self.col_right.set_data([ R3,  R3], [0.0, H - r2])

        # --- recompute spring polylines with new geometry ---
        Xl, Yl = self._spring_polyline(
            (0-R3, self.cfg.spring_start), (0-R3, self.cfg.spring_start+self.cfg.spring_length),
            coils=self.cfg.spring_coils, amp=self.cfg.spring_amp
        )
        Xr, Yr = self._spring_polyline(
            (0+R3, self.cfg.spring_start), (0+R3, self.cfg.spring_start+self.cfg.spring_length),
            coils=self.cfg.spring_coils, amp=self.cfg.spring_amp
        )
        self.spring_left.set_data(Xl, Yl)
        self.spring_right.set_data(Xr, Yr)

        # Recompute anim limits if desired (optional; keep fixed for speed)
        # span = max(L, H) + R3 + 0.05
        # self.ax_anim.set_xlim(-span, span)
        # self.ax_anim.set_ylim(-span, span)

        # Re-capture backgrounds after static layout changes
        self._init_blit()

    def update(self, obs, action, goal, t: float):
        # ---- Unpack obs ----
        theta = float(obs[0])
        theta_goal = float(goal[0])
        phi1  = float(obs[2]) 
        phi2  = float(obs[4])
        coact_goal = float(goal[1])
        tau1 = float(action[0])
        tau2 = float(action[1])

        # ---- Move link (read params live; theta=0 points DOWN) ----
        L = float(self.p.l)
        x_tip = L * np.sin(theta)
        y_tip = -L * np.cos(theta)
        self.link_ln.set_data([0.0, x_tip], [0.0, y_tip])

        # ---- Ring-buffer append; fetch latest N samples ----
        last_i = self._rb_append(t, theta, theta_goal, coact_goal, phi1, phi2, tau1, tau2)
        T, Th, Th_goal, P1, P2, coact, coact_goal, tau1, tau2 = self._rb_latestN(last_i)

        # ---- Plot latest N with relative time so x-lims stay fixed ----
        if T.size:
            X = T - T[-1]   # seconds in [-win, 0]
            self.ln_theta.set_data(X, Th)
            self.ln_theta_goal.set_data(X, Th_goal)
            self.ln_phi1.set_data(X, P1)
            self.ln_phi2.set_data(X, P2)
            self.ln_coact.set_data(X, coact)
            self.ln_coact_goal.set_data(X, coact_goal)
            self.ln_tau1.set_data(X, tau1)
            self.ln_tau2.set_data(X, tau2)
        else:
            self.ln_theta.set_data([], [])
            self.ln_theta_goal.set_data([], [])
            self.ln_coact_goal.set_data([], [])
            self.ln_phi1.set_data([], [])
            self.ln_phi2.set_data([], [])
            self.ln_coact.set_data([], [])
            self.ln_tau1.set_data([], [])
            self.ln_tau2.set_data([], [])

        # ---- Blit only changed artists ----
        c = self.fig.canvas
        c.restore_region(self._bg_anim);  self.ax_anim.draw_artist(self.link_ln);  c.blit(self.ax_anim.bbox)
        c.restore_region(self._bg_theta); self.ax_theta.draw_artist(self.ln_theta); self.ax_theta.draw_artist(self.ln_theta_goal); c.blit(self.ax_theta.bbox)
        c.restore_region(self._bg_phi1);  self.ax_phi1.draw_artist(self.ln_phi1);  c.blit(self.ax_phi1.bbox)
        c.restore_region(self._bg_phi2);  self.ax_phi2.draw_artist(self.ln_phi2);  c.blit(self.ax_phi2.bbox)
        c.restore_region(self._bg_coact);  self.ax_coact.draw_artist(self.ln_coact); self.ax_theta.draw_artist(self.ln_coact_goal); c.blit(self.ax_coact.bbox)
        c.restore_region(self._bg_tau1);  self.ax_tau1.draw_artist(self.ln_tau1);  c.blit(self.ax_tau1.bbox)
        c.restore_region(self._bg_tau2);  self.ax_tau2.draw_artist(self.ln_tau2);  c.blit(self.ax_tau2.bbox)
        c.flush_events()

    def close(self):
        plt.close(self.fig)

    # ---------- internals ----------
    def _build_static_anim(self):
        """Create static animation elements once from current params."""
        R3 = float(self.p.r3)   # big pulley radius
        L  = float(self.p.l)    # link length
        r1 = float(self.p.r1); r2 = float(self.p.r2)
        H  = L * self.cfg.H_scale

        c_big = (0.0, 0.0)
        c_p1  = (-(r1*2), H)
        c_p2  = (r2*2, H)

        self.wheel_big = Circle(c_big, R3, fill=False, lw=2, antialiased=False)
        self.wheel_p1  = Circle(c_p1,  r1, fill=False, lw=1.5, ls="--", antialiased=False)
        self.wheel_p2  = Circle(c_p2,  r2, fill=False, lw=1.5, ls="--", antialiased=False)
        self.ax_anim.add_patch(self.wheel_big)
        self.ax_anim.add_patch(self.wheel_p1)
        self.ax_anim.add_patch(self.wheel_p2)

        # columns & axles (keep references so we can update in refresh_static)
        (self.col_left,)  = self.ax_anim.plot([-R3, -R3], [0.0, H], lw=1, color="black", antialiased=False)
        (self.col_right,) = self.ax_anim.plot([ R3,  R3], [0.0, H], lw=1, color="black", antialiased=False)
        self.ax_anim.plot(*c_big, marker="o", ms=4)
        self.ax_anim.plot(*c_p1, marker="o", ms=3)
        self.ax_anim.plot(*c_p2, marker="o", ms=3)

        # link (animated; data updated each frame)
        (self.link_ln,) = self.ax_anim.plot([0.0, 0.0], [0.0, -L], lw=2.0, zorder=5, antialiased=True)
        self.ax_anim.set_title("Pulley 3 + link")

        # --- springs: left center and right center (static artists) ---
        Xl, Yl = self._spring_polyline(
            (0-R3, self.cfg.spring_start), (0-R3, self.cfg.spring_start+self.cfg.spring_length),
            coils=self.cfg.spring_coils, amp=self.cfg.spring_amp
        )
        Xr, Yr = self._spring_polyline(
            (0+R3, self.cfg.spring_start), (0+R3, self.cfg.spring_start+self.cfg.spring_length),
            coils=self.cfg.spring_coils, amp=self.cfg.spring_amp
        )
        (self.spring_left,) = self.ax_anim.plot(
            Xl, Yl, lw=self.cfg.spring_lw, color="black", antialiased=False, zorder=3
        )
        (self.spring_right,) = self.ax_anim.plot(
            Xr, Yr, lw=self.cfg.spring_lw, color="black", antialiased=False, zorder=3
        )

        # keep anim axes limits fixed for speed
        span = max(L, H) + R3 + 0.05
        self.ax_anim.set_xlim(-span, span)
        self.ax_anim.set_ylim(-span, span)

    def _init_blit(self):
        """Render static elements once, then capture backgrounds for blitting."""
        self.fig.canvas.draw()
        c = self.fig.canvas
        self._bg_anim  = c.copy_from_bbox(self.ax_anim.bbox)
        self._bg_theta = c.copy_from_bbox(self.ax_theta.bbox)
        self._bg_phi1  = c.copy_from_bbox(self.ax_phi1.bbox)
        self._bg_phi2  = c.copy_from_bbox(self.ax_phi2.bbox)
        self._bg_coact  = c.copy_from_bbox(self.ax_coact.bbox)
        self._bg_tau1  = c.copy_from_bbox(self.ax_tau1.bbox)
        self._bg_tau2  = c.copy_from_bbox(self.ax_tau2.bbox)

    def _on_resize(self, *_):
        # Re-capture backgrounds after a resize so blitting remains valid
        self._init_blit()

    def _rb_append(self, t, th, theta_goal, coact_goal, p1, p2, tau1, tau2):
        i = self.t_idx % self.cfg.N
        self.t_buf[i]  = t
        self.th_buf[i] = th
        self.th_goal_buf[i] = theta_goal
        self.coact_goal_buf[i] = coact_goal
        self.p1_buf[i] = p1
        self.p2_buf[i] = p2
        self.coact_buf[i] = (tau1+tau2)/self.p.r2
        self.tau1_buf[i] = tau1
        self.tau2_buf[i] = tau2
        self.t_idx += 1
        return i

    def _rb_latestN(self, last_i):
        """Return last <=N samples in time order (small arrays)."""
        N = self.cfg.N
        if self.t_idx < N:
            T  = self.t_buf[:self.t_idx]
            Th = self.th_buf[:self.t_idx]
            Th_goal = self.th_goal_buf[:self.t_idx]
            P1 = self.p1_buf[:self.t_idx]
            P2 = self.p2_buf[:self.t_idx]
            coact = self.coact_buf[:self.t_idx]
            coact_goal = self.coact_goal_buf[:self.t_idx]
            tau1 = self.tau1_buf[:self.t_idx]
            tau2 = self.tau2_buf[:self.t_idx]
        else:
            if last_i + 1 < N:
                T  = np.concatenate((self.t_buf[last_i+1:],  self.t_buf[:last_i+1]))
                Th = np.concatenate((self.th_buf[last_i+1:], self.th_buf[:last_i+1]))
                Th_goal = np.concatenate((self.th_goal_buf[last_i+1:], self.th_goal_buf[:last_i+1]))
                P1 = np.concatenate((self.p1_buf[last_i+1:], self.p1_buf[:last_i+1]))
                P2 = np.concatenate((self.p2_buf[last_i+1:], self.p2_buf[:last_i+1]))
                coact = np.concatenate((self.coact_buf[last_i+1:], self.coact_buf[:last_i+1]))
                coact_goal = np.concatenate((self.coact_goal_buf[last_i+1:], self.coact_goal_buf[:last_i+1]))
                tau1 = np.concatenate((self.tau1_buf[last_i+1:], self.tau1_buf[:last_i+1]))
                tau2 = np.concatenate((self.tau2_buf[last_i+1:], self.tau2_buf[:last_i+1]))
            else:
                # exactly filled, no wrap
                T, Th, Th_goal, P1, P2, coact, coact_goal, tau1, tau2 = (self.t_buf, self.th_buf, self.th_goal_buf, self.p1_buf, self.p2_buf, 
                                                    self.coact_buf, self.coact_goal_buf, self.tau1_buf, self.tau2_buf)
        return T, Th, Th_goal, P1, P2, coact, coact_goal, tau1, tau2
    
    def _spring_polyline(self, start, end, coils: int, amp: float):
        """
        Build a zig-zag spring polyline from 'start' to 'end'.
        'start' and 'end' are 2D points in world coords (iterables of length 2).
        Endpoints are NOT offset (the spring meets the endpoints exactly).

        Returns (X, Y) arrays.
        """
        p0 = np.asarray(start, float)
        p1 = np.asarray(end, float)

        # Direction & length
        d = p1 - p0
        L = np.linalg.norm(d)
        if L <= 1e-9:
            # Degenerate; return a dot
            return np.array([p0[0], p1[0]]), np.array([p0[1], p1[1]])

        u = d / L                       # unit tangent
        n = np.array([-u[1], u[0]])     # unit normal (left-handed)

        # Parameterization along the line
        k = max(1, coils) * 2           # peaks + valleys
        ts = np.linspace(0.0, 1.0, k + 1)

        # Alternate +amp / -amp; keep endpoints un-offset
        signs = np.where(np.arange(k + 1) % 2 == 0, 1.0, -1.0)
        signs[0] = 0.0
        signs[-1] = 0.0

        # Safety clamp so amplitude isn't ridiculous vs length
        a = min(amp, 0.25 * L)

        pts = p0[None, :] + ts[:, None] * d[None, :] + (signs[:, None] * a) * n[None, :]
        X, Y = pts[:, 0], pts[:, 1]
        return X, Y
    
    def clear_buffers(self):
        """Clear time buffers and empty the plotted lines (used on env reset)."""
        self.t_idx = 0
        self.t_buf.fill(0.0); self.th_buf.fill(0.0)
    
        self.p1_buf.fill(0.0); self.p2_buf.fill(0.0)
        self.tau1_buf.fill(0.0); self.tau2_buf.fill(0.0)
        # empty the time-series lines
        self.ln_theta.set_data([], [])
        self.ln_theta_goal.set_data([], [])
        self.ln_coact_goal.set_data([], [])
        self.ln_phi1.set_data([], [])
        self.ln_phi2.set_data([], [])
        self.ln_coact.set_data([], [])
        self.ln_tau1.set_data([], [])
        self.ln_tau2.set_data([], [])
        # reset link to current theta=0 pose (or keep last; your call)
        L = float(self.p.l)
        self.link_ln.set_data([0.0, 0.0], [0.0, -L])
        # re-capture backgrounds to avoid ghosting after a big jump
        self._init_blit()
        # draw once
        c = self.fig.canvas
        c.restore_region(self._bg_anim);  self.ax_anim.draw_artist(self.link_ln);  c.blit(self.ax_anim.bbox)
        c.restore_region(self._bg_theta); self.ax_theta.draw_artist(self.ln_theta); self.ax_theta.draw_artist(self.ln_theta_goal); c.blit(self.ax_theta.bbox)
        c.restore_region(self._bg_phi1);  self.ax_phi1.draw_artist(self.ln_phi1);  c.blit(self.ax_phi1.bbox)
        c.restore_region(self._bg_phi2);  self.ax_phi2.draw_artist(self.ln_phi2);  c.blit(self.ax_phi2.bbox)
        c.restore_region(self._bg_coact);  self.ax_coact.draw_artist(self.ln_coact); self.ax_theta.draw_artist(self.ln_coact_goal); c.blit(self.ax_coact.bbox)
        c.restore_region(self._bg_tau1);  self.ax_tau1.draw_artist(self.ln_tau1);  c.blit(self.ax_tau1.bbox)
        c.restore_region(self._bg_tau2);  self.ax_tau2.draw_artist(self.ln_tau2);  c.blit(self.ax_tau2.bbox)
        c.flush_events()
