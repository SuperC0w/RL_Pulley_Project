from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox


@dataclass
class ActionState2D:
    tau1: float = 0.0
    tau2: float = 0.0
    tau3: float = 0.0

    def as_array(self):
        return np.array([self.tau1, self.tau2, self.tau3], dtype=float)


class ControlPanel2D:
    """
    Control panel for 3 torques (tau1/2/3) and two angular goals (theta1/theta2).
    - Action sliders update an ActionState you read each step.
    - Parameter sliders mutate the provided `params` object directly (live).
    """

    def __init__(self, params, tau1_lim=0.2, tau2_lim=0.2, tau3_lim=0.2, F_range=(-2.0, 2.0),
                 show_param_sliders=True, on_reset=None, on_impulse=None, on_goal_mode_change=None):
        self.params = params
        self.actions = ActionState2D()
        self.on_reset = on_reset
        self.on_impulse = on_impulse
        self.on_goal_mode_change = on_goal_mode_change
        self.impulsed_triggered_flag = False
        self.follow_trajectory_flag = False

        self.fig, self.ax = plt.subplots(figsize=(5.4, 6.6))
        self.fig.canvas.manager.set_window_title("Pulley Controls 2D")
        self.ax.set_axis_off()

        y = 0.95
        self._add_label("Actions", y); y -= 0.06
        self.s_tau1 = self._add_slider("\u03c4$_1$ (Nm)", -tau1_lim, tau1_lim, 0.0, y); y -= 0.06
        self.s_tau2 = self._add_slider("\u03c4$_2$ (Nm)", -tau2_lim, tau2_lim, 0.0, y); y -= 0.06
        self.s_tau3 = self._add_slider("\u03c4$_3$ (Nm)", -tau3_lim, tau3_lim, 0.0, y); y -= 0.06

        btns = self._add_button_row(["Zero actions", "Reset simulation", "Force Impulse"], y)
        self.b_zero, self.b_reset, self.b_impulse = btns
        self.b_zero.on_clicked(self._zero_actions)
        self.b_reset.on_clicked(self._on_reset_clicked)
        self.b_impulse.on_clicked(self._on_impulse_clicked)
        y -= 0.05

        self._add_label("Goal Configuration", y); y -= 0.05
        ax_goal_mode = self.fig.add_axes([0.10, y, 0.40, 0.025]); ax_goal_mode.set_axis_off()
        self.chk_goal_mode = CheckButtons(ax_goal_mode, ["Follow trajectory"], [self.follow_trajectory_flag])
        self.chk_goal_mode.on_clicked(self._on_goal_mode_toggled)
        y -= 0.06

        # Theta goal slider 
        self.s_theta1_goal = self._add_slider("Goal \u03B8$_1$ (rad)", -np.pi/2, np.pi/2, 0.0, y); y -= 0.06
        self.s_theta2_goal = self._add_slider("Goal \u03B8$_2$ (rad)", -np.pi/2, np.pi/2, 0.0, y); y -= 0.06
        
        # Coactivation goal slider 
        self.s_coact_goal = self._add_slider("Coact Torque (N)", (0 + 0.02/2 + 0.02/4)*2, params.tau_max1 - 0.020/2, 0.050, y); y -= 0.06

        amp_slider_rect = [0.10, y, 0.80, 0.04]; amp_text_rect = [0.90, y, 0.095, 0.04]
        self.goal_amplitude = self._add_slider("Amplitude (rad)", 0.0, np.pi, 0.1, y, rect=amp_slider_rect)
        self.goal_amplitude.valtext.set_visible(False)
        self.tb_amp = self._add_textbox("", amp_text_rect, initial=f"{self.goal_amplitude.val:.3f}")
        y -= 0.06

        def _amp_slider_to_tb(val):
            try:
                self.tb_amp.set_val(f"{float(val):.3f}")
            except Exception:
                pass
        self.goal_amplitude.on_changed(_amp_slider_to_tb)

        def _amp_tb_to_slider(text):
            try:
                v = float(text)
            except Exception:
                self.tb_amp.set_val(f"{self.goal_amplitude.val:.3f}")
                return
            v = max(self.goal_amplitude.valmin, min(self.goal_amplitude.valmax, v))
            self.goal_amplitude.set_val(v)
        self.tb_amp.on_submit(_amp_tb_to_slider)

        self.goal_period = self._add_slider("Period (s)", 0.1, 5.0, 1.0, y); y -= 0.08

        self._set_slider(self.goal_amplitude, self.follow_trajectory_flag)
        self._set_slider(self.goal_period, self.follow_trajectory_flag)
        self._set_textbox(self.tb_amp, self.follow_trajectory_flag)

        if show_param_sliders:
            self._add_label("Params (live)", y); y -= 0.05
            self.s_F = self._add_slider("F (N)", F_range[0], F_range[1], float(self.params.F), y); y -= 0.08
            self.s_F.on_changed(lambda v: self._set_attr("F", float(v)))

        self.s_tau1.on_changed(lambda v: self._set_action("tau1", float(v)))
        self.s_tau2.on_changed(lambda v: self._set_action("tau2", float(v)))
        self.s_tau3.on_changed(lambda v: self._set_action("tau3", float(v)))

        self.fig.tight_layout()

    # ---------- helpers ----------
    def _add_label(self, text, y):
        self.fig.text(0.10, y, text, fontsize=11, weight="bold")

    def _add_slider(self, label, vmin, vmax, vinit, y, rect=None, label_top=True):
        rect = rect or [0.10, y, 0.80, 0.04]
        ax = self.fig.add_axes(rect)
        s = Slider(ax=ax, label=label, valmin=vmin, valmax=vmax, valinit=vinit)
        if label_top:
            lbl = s.label; lbl.set_clip_on(False); lbl.set_ha("center"); lbl.set_va("bottom")
            lbl.set_transform(ax.transAxes); lbl.set_position((0.5, 0.8))
        return s

    def _add_button(self, label, y):
        ax = self.fig.add_axes([0.10, y, 0.80, 0.05])
        return Button(ax=ax, label=label)

    def _add_textbox(self, label, rect, initial=""):
        ax = self.fig.add_axes(rect)
        tb = TextBox(ax, label=label, initial=initial)
        return tb

    def _set_action(self, name, value):
        setattr(self.actions, name, value)

    def _set_attr(self, name, value):
        setattr(self.params, name, value)

    def _set_slider(self, slider, enabled):
        slider.set_active(enabled)
        color = "black" if enabled else "0.5"; alpha = 1.0 if enabled else 0.3
        slider.label.set_color(color)
        slider.valtext.set_color(color)
        slider.track.set_alpha(alpha)
        slider.poly.set_alpha(alpha)
        slider._handle.set_alpha(alpha)
        slider.ax.figure.canvas.draw_idle()

    def _set_textbox(self, tb, enabled):
        tb.set_active(enabled)
        txt_color = "black" if enabled else "0.5"; alpha = 1.0 if enabled else 0.3
        tb.text_disp.set_color(txt_color)
        tb.label.set_color(txt_color)
        tb.ax.patch.set_alpha(alpha)
        tb.ax.figure.canvas.draw_idle()

    def _add_button_row(self, labels, y, left=0.10, right=0.90, height=0.05, gap=0.02):
        n = len(labels); total_gap = gap * (n - 1); width = (right - left - total_gap); width_per = width / n
        buttons = []
        for i, lab in enumerate(labels):
            x = left + i * (width_per + gap)
            ax_btn = self.fig.add_axes([x, y, width_per, height])
            buttons.append(Button(ax=ax_btn, label=lab))
        return buttons

    def _zero_actions(self, _event):
        self.actions = ActionState2D()
        self.s_tau1.set_val(0.0); self.s_tau2.set_val(0.0); self.s_tau3.set_val(0.0)

    def _on_reset_clicked(self, _event):
        if self.on_reset:
            self.on_reset()

    def _on_impulse_clicked(self, _event):
        if not self.impulsed_triggered_flag:
            self.impulsed_triggered_flag = True
        if self.on_impulse:
            self.on_impulse()

    def _on_goal_mode_toggled(self, _label):
        self.follow_trajectory_flag = not self.follow_trajectory_flag
        self._set_slider(self.s_theta1_goal, not self.follow_trajectory_flag)
        self._set_slider(self.s_theta2_goal, not self.follow_trajectory_flag)
        self._set_slider(self.goal_amplitude, self.follow_trajectory_flag)
        self._set_slider(self.goal_period, self.follow_trajectory_flag)
        self._set_textbox(self.tb_amp, self.follow_trajectory_flag)
        if self.on_goal_mode_change:
            self.on_goal_mode_change()
        self.fig.canvas.draw_idle()
