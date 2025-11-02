from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

@dataclass
class ActionState:
    tau1: float = 0.0
    tau2: float = 0.0
    def as_array(self):
        # Only tau1 and tau2 are actions
        return np.array([self.tau1, self.tau2], dtype=float)

class ControlPanel:
    """
    A second window with sliders for actions (tau1, tau2) and parameters (e.g., F, k1, k2, c1, c2, l).
    - Action sliders update an ActionState you read each step.
    - Parameter sliders mutate the provided `params` object directly (live).

    Usage:
        panel = ControlPanel(params, tau1_lim=2.0, tau2_lim=2.0, F_range=(-20.0, 20.0))
        ...
        action = panel.actions.as_array()  # -> array([tau1, tau2])
        # params.F (and others) are updated live by the sliders
    """
    def __init__(self, params, tau1_lim=2.0, tau2_lim=2.0, F_range=(-20.0, 20.0),
                 show_param_sliders=True, on_reset=None, on_impulse=None, on_goal_mode_change=None):
        self.params = params
        self.actions = ActionState()
        self.on_reset = on_reset
        self.on_impulse = on_impulse
        self.on_goal_mode_change = on_goal_mode_change
        self.impulsed_triggered_flag = False
        self.follow_trajectory_flag = False

        # ---- Figure layout ----
        self.fig, self.ax = plt.subplots(figsize=(5.2, 6.4))
        self.fig.canvas.manager.set_window_title("Pulley Controls")
        self.ax.set_axis_off()

        y = 0.95
        self._add_label("Actions", y); y -= 0.06

        # Action sliders (tau1, tau2)
        self.s_tau1 = self._add_slider("τ1 (N·m)",  -tau1_lim, tau1_lim, 0.0, y); y -= 0.06
        self.s_tau2 = self._add_slider("τ2 (N·m)",  -tau2_lim, tau2_lim, 0.0, y); y -= 0.06

        # Creating buttons
        btns = self._add_button_row(["Zero actions", "Reset simulation", "Force Impulse"], y)
        self.b_zero, self.b_reset, self.b_impulse = btns
        self.b_zero.on_clicked(self._zero_actions)
        self.b_reset.on_clicked(self._on_reset_clicked)
        self.b_impulse.on_clicked(self._on_impulse_clicked)
        y -= 0.05

        self._add_label("Goal Configuration", y); y -= 0.05
        
        # CheckBox to enable/disable trajectory tracking
        ax_goal_mode = self.fig.add_axes([0.10, y, 0.40, 0.025])
        ax_goal_mode.set_axis_off()
        self.chk_goal_mode = CheckButtons(ax_goal_mode, ["Follow trajectory"], [self.follow_trajectory_flag])
        self.chk_goal_mode.on_clicked(self._on_goal_mode_toggled)
        y -= 0.06

        # Theta goal slider 
        self.s_theta_goal = self._add_slider("Goal θ (radians)",  -np.pi/2, np.pi/2, 0.0, y); y -= 0.06
        
        # Coactivation goal slider 
        self.coact_goal = self._add_slider("Coact Force (N)",  0, 8.6, 0.0, y); y -= 0.06

        # Trajectory amplitude
        amp_slider_rect = [0.10, y, 0.80, 0.04]
        amp_text_rect   = [0.90, y, 0.095, 0.04]
        self.goal_amplitude = self._add_slider("Amplitude (radians)", 0, np.pi/2, 0.0393, y, rect=amp_slider_rect)
        self.goal_amplitude.valtext.set_visible(False)
        self.tb_amp = self._add_textbox("", amp_text_rect, initial=f"{self.goal_amplitude.val:.4f}")
        y -= 0.06

        def _amp_slider_to_tb(val):
            try:
                self.tb_amp.set_val(f"{float(val):.4f}")
            except Exception:
                pass
        self.goal_amplitude.on_changed(_amp_slider_to_tb)

        def _amp_tb_to_slider(text):
            try:
                v = float(text)
            except Exception:
                self.tb_amp.set_val(f"{self.goal_amplitude.val:.4f}")
                return
            v = max(self.goal_amplitude.valmin, min(self.goal_amplitude.valmax, v))
            self.goal_amplitude.set_val(v)
        self.tb_amp.on_submit(_amp_tb_to_slider)

        # Trajectory period 
        self.goal_period = self._add_slider("Period (s)",  0.1, 1, 0.1, y); y -= 0.06

        self._set_slider(self.goal_amplitude, self.follow_trajectory_flag)
        self._set_slider(self.goal_period, self.follow_trajectory_flag)
        self._set_textbox(self.tb_amp, self.follow_trajectory_flag)

        # Optional parameter sliders
        if show_param_sliders:
            self._add_label("Params (live)", y); y -= 0.05

            # F as a PARAMETER (live), not an action:
            self.s_F  = self._add_slider("F (N)", F_range[0], F_range[1], float(self.params.F), y); y -= 0.06

            # Example other parameters
            self.s_k1 = self._add_slider("k1 (N/m)",    20.0, 1000.0, float(self.params.k1), y); y -= 0.06
            self.s_k2 = self._add_slider("k2 (N/m)",    20.0, 1000.0, float(self.params.k2), y); y -= 0.06
            self.s_c1 = self._add_slider("c1 (N·m·s)",     6.9e-8,   15.0e-5, float(self.params.c1), y); y -= 0.06
            self.s_c2 = self._add_slider("c2 (N·m·s)",     6.9e-8,   15.0e-5, float(self.params.c2), y); y -= 0.06
            self.s_l  = self._add_slider("l (m)",         0.02,    0.50, float(self.params.l),  y); y -= 0.10

            # Hook PARAM changes to the live params object
            self.s_F .on_changed(lambda v: self._set_attr("F",  float(v)))
            self.s_k1.on_changed(lambda v: self._set_attr("k1", float(v)))
            self.s_k2.on_changed(lambda v: self._set_attr("k2", float(v)))
            self.s_c1.on_changed(lambda v: self._set_attr("c1", float(v)))
            self.s_c2.on_changed(lambda v: self._set_attr("c2", float(v)))
            self.s_l .on_changed(lambda v: self._set_attr("l",  float(v)))

        # Hook ACTIONS
        self.s_tau1.on_changed(lambda v: self._set_action("tau1", float(v)))
        self.s_tau2.on_changed(lambda v: self._set_action("tau2", float(v)))

        self.fig.tight_layout()

    # ---------- helpers ----------
    def _add_label(self, text, y):
        self.fig.text(0.10, y, text, fontsize=11, weight="bold")

    def _add_slider(self, label, vmin, vmax, vinit, y, rect=None, label_top=True):
        if rect is None:
            rect = [0.10, y, 0.80, 0.04]  # left, bottom, width, height in figure coords
        ax = self.fig.add_axes(rect)
        s = Slider(ax=ax, label=label, valmin=vmin, valmax=vmax, valinit=vinit)

        # Avoid clipping and optionally move label above
        if label_top:
            lbl = s.label
            lbl.set_clip_on(False)
            lbl.set_ha("center")
            lbl.set_va("bottom")
            lbl.set_transform(ax.transAxes)     # use axes coords (0..1)
            lbl.set_position((0.5, 0.8))       # centered, a bit above (tweak 1.10–1.30)

        return s

    def _add_button(self, label, y):
        ax = self.fig.add_axes([0.10, y, 0.80, 0.05])
        return Button(ax=ax, label=label)
    
    def _add_textbox(self, label, rect, initial=""):
        ax = self.fig.add_axes(rect)
        tb = TextBox(ax, label=label, initial=initial)
        # if getattr(tb, "label", None) is not None:
        #     tb.label.set_clip_on(False)
        return tb

    def _set_action(self, name, value):
        setattr(self.actions, name, value)

    def _set_attr(self, name, value):
        # Mutate params live. If your dynamics capture `params` by reference,
        # changes take effect next step without rebuilding anything.
        setattr(self.params, name, value)

    def _set_slider(self, slider, enabled):
        """
        Disable or enable slider
        """
        # Interaction
        slider.set_active(enabled)

        # Set visuals
        if enabled:
            color = 'black'
            alpha = 1.0
        else:
            color = '0.5'
            alpha = 0.3
        slider.label.set_color(color)
        slider.valtext.set_color(color)
        slider.track.set_alpha(alpha)
        slider.poly.set_alpha(alpha)
        slider._handle.set_alpha(alpha)
        
        # Redraw slider
        slider.ax.figure.canvas.draw_idle()

    def _set_textbox(self, tb, enabled):
        tb.set_active(enabled)

        if enabled:
            txt_color = 'black'
            alpha = 1.0
        else:
            txt_color = '0.5'
            alpha = 0.3
        tb.text_disp.set_color(txt_color)
        tb.label.set_color(txt_color)
        tb.ax.patch.set_alpha(alpha)
        
        tb.ax.figure.canvas.draw_idle()

    def _add_button_row(self, labels, y, left=0.10, right=0.90, height=0.05, gap=0.02):
        """
        Create a row of equally-spaced buttons.
        labels: list[str]
        y: bottom position in figure coords
        left/right/height/gap: layout params in figure coords
        Returns: list[Button]
        """
        n = len(labels)
        assert n >= 1, "Must have at least one label"
        total_gap = gap * (n - 1)
        width = (right - left - total_gap)
        width_per = width / n
        buttons = []
        for i, lab in enumerate(labels):
            x = left + i * (width_per + gap)
            ax_btn = self.fig.add_axes([x, y, width_per, height])
            buttons.append(Button(ax=ax_btn, label=lab))
        return buttons

    def _zero_actions(self, _event):
        self.actions = ActionState()
        self.s_tau1.set_val(0.0)
        self.s_tau2.set_val(0.0)

    def _on_reset_clicked(self, _event):
        # zero actions visually & internally
        # self._zero_actions()
        # call user-supplied reset (env + renderer) if provided
        if self.on_reset:
            self.on_reset()

    def _on_impulse_clicked(self, _event):
        if self.impulsed_triggered_flag == False:
            self.impulsed_triggered_flag = True
        if self.on_impulse:
            self.on_impulse()

    def _on_goal_mode_toggled(self, _label):
        self.follow_trajectory_flag = not self.follow_trajectory_flag

        # Set theta goal slider inactive
        self._set_slider(self.s_theta_goal, not self.follow_trajectory_flag)
        self._set_slider(self.goal_amplitude, self.follow_trajectory_flag)
        self._set_slider(self.goal_period, self.follow_trajectory_flag)
        self._set_textbox(self.tb_amp,      self.follow_trajectory_flag)

        if self.on_goal_mode_change: 
            self.on_goal_mode_change()
        self.fig.canvas.draw_idle()
