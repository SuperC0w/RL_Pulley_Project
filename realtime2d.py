import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

from env2d.params import PulleyParams
from env2d.pulley_env import PulleyEnv
from ui.render_matplotlib2d import MatplotlibRenderer2D
from ui.controls2d import ControlPanel2D

def main():
    parser = argparse.ArgumentParser(description="Pulley control simulation")
    parser.add_argument("-c", "--controller", choices=["pid", "model", "manual"], default="manual",
                        help="Select controller type: pid | model | manual")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument("--fps", type=int, default=90, help="Render frames per second")
    args = parser.parse_args()
    use_pid = args.controller == "pid"
    use_model = args.controller == "model"

    dt = args.dt
    render_dt = 1.0 / args.fps

    params = PulleyParams()
    env = PulleyEnv(params, dt=dt, max_steps=100_000, seed=0)
    renderer = MatplotlibRenderer2D(params, dt_sim=env.dt)
    renderer.draw_static()

    def do_reset():
        env.reset()
        renderer.clear_buffers()

    panel = ControlPanel2D(
        params,
        tau1_lim=params.tau_max1,
        tau2_lim=params.tau_max2,
        tau3_lim=params.tau_max3,
        F_range=(-2.0, 2.0),
        show_param_sliders=True,
        on_reset=do_reset,
        on_impulse=env.trigger_impulse,
    )

    plt.ion()
    plt.show(block=False)
    renderer.fig.canvas.draw()
    renderer.fig.canvas.flush_events()

    obs, info = env.reset()
    
    # track starting time of script
    start_time = time.perf_counter()
    count = 0

    last = time.perf_counter()
    acc = 0.0   # accumulator of real time
    current_time = 0.0
    last_render = last
    MAX_ACC = 0.25
    try:
        while True:
            if not plt.fignum_exists(renderer.fig.number):
                break

            now = time.perf_counter()
            acc += now - last
            last = now
            if acc > MAX_ACC:
                print(count)
                print("dropping frames")
                acc = MAX_ACC   # drop sim frames if we fell behind badly

            while acc >= dt:
                count += 1
                if panel.follow_trajectory_flag:
                    omega = 2 * np.pi / max(panel.goal_period.val, 1e-3)
                    theta1_goal = panel.goal_amplitude.val * np.sin(omega * current_time)
                    theta2_goal = panel.goal_amplitude.val * np.sin(omega * current_time)
                else:
                    theta1_goal = panel.s_theta1_goal.val
                    theta2_goal = panel.s_theta2_goal.val

                # impulse handled directly via on_impulse; clear flag if set
                if panel.impulsed_triggered_flag:
                    panel.impulsed_triggered_flag = False

                action = panel.actions.as_array()
                #TESTING->CHECKING MODEL
                action = np.array([0.129/2, 0.129/2, 0.129])
                obs, info = env.step(action)
                current_time = info.get("t", current_time)
                acc -= dt

            if now - last_render >= render_dt:
                renderer.update(obs, action, t=current_time)
                last_render = now
    except KeyboardInterrupt:
        pass
    finally:
        end_time = time.perf_counter()
        renderer.close()
        print(f"number of cycles: {count}, elapsed time: {end_time - start_time:.3f}")

if __name__ == "__main__":
    main()
