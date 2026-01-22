import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

from env.params import PulleyParams
from env.pulley_env import PulleyEnv
from ui.render_matplotlib import MatplotlibRenderer
from ui.controls import ControlPanel
from controllers.pd import PD_1d
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.gym_env import PulleyEnvGym
# from env.gym_env_single_action_vel_ctrl import PulleyEnvGym

from stable_baselines3 import SAC

def _wrap_pi(x):
        """
        Wrap to (-pi, pi]
        """
        return (x + np.pi) % (2*np.pi) - np.pi

def _map_reparam_to_torques(a: np.ndarray, coact_goal, pulley_radius) -> np.ndarray:
        """
        Map reparameterized action [u_coact (N), delta (Nm)] to torques [tau1, tau2] (Nm),
        enforcing bounds so both torques are within [0, u_max] and |tau1 - tau2| <= 2*|delta|.
        Used for model
        """
        u_coact = coact_goal
        delta = float(a[0])

        # Base torque from coactivation: tau1 + tau2 = u_coact * r1, select base such that tau1 = tau2
        base = 0.5 * u_coact * float(pulley_radius)

        tau1 = base + delta/2
        tau2 = base - delta/2

        return np.array([tau1, tau2], dtype=np.float32)

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

    env_params = PulleyParams()
    env = PulleyEnv(env_params, dt=dt, max_steps=100_000, seed=0)
    action = np.zeros(2, dtype=np.float32)

    renderer = MatplotlibRenderer(env_params, dt_sim=env.dt)
    renderer.draw_static()

    if use_pid and use_model:
        raise ValueError("Both USE_PID and USE_MODEL cannot be true at the same time")

    if use_pid:
        pid_controller = PD_1d()
    elif use_model:
        # model = SAC.load("./models/SAC/best/best_model.zip", device="cpu")
        # vecnorm_path = "./models/SAC/best/best_vecnorm.pkl"
        # model = SAC.load("./models/SAC/best/current_best.zip", device="cpu")
        # vecnorm_path = "./models/SAC/best/current_best.pkl"
        model = SAC.load("./models/SAC/best/perf2.zip", device="cpu")
        vecnorm_path = "./models/SAC/best/perf1.pkl"

        venv_for_norm = DummyVecEnv([lambda: PulleyEnvGym(dt=dt, max_steps=100_000, seed=0)])
        vecnorm = VecNormalize.load(vecnorm_path, venv_for_norm)
        obs_mean = vecnorm.obs_rms.mean.copy()
        obs_var  = vecnorm.obs_rms.var.copy()
        clip_obs = float(vecnorm.clip_obs)
        del vecnorm, venv_for_norm  # free it

        def normalize_obs(obs_vec: np.ndarray) -> np.ndarray:
            # VecNormalize uses: clip((obs - mean) / sqrt(var + eps), -clip, clip)
            eps = 1e-8
            normed = (obs_vec - obs_mean) / np.sqrt(obs_var + eps)
            return np.clip(normed, -clip_obs, clip_obs)

    def do_reset():
        env.reset()
        renderer.clear_buffers()

    panel = ControlPanel(
        env_params,
        tau1_lim=env_params.tau_max,
        tau2_lim=env_params.tau_max,
        F_range=(-2.0, 2.0),
        show_param_sliders=True,
        on_reset=do_reset,
        on_impulse=env.trigger_impulse
    )

    plt.ion()
    plt.show(block=False)
    renderer.fig.canvas.draw()
    renderer.fig.canvas.flush_events()

    obs, info = env.reset()
    print("Controls are in the separate 'Pulley Controls' window.")

    hold_step_count = 9 # number of steps to hold before triggering force impulse

    # track starting time of script
    start_time = time.perf_counter()
    count = 0

    last = time.perf_counter()
    acc = 0.0   # accumulator of real time
    current_time = 0.0

    theta_goal = 0
    coact_goal = 0

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
                hold_step_count += 1

                # get goal value from control panel
                if use_pid or use_model:
                    if panel.follow_trajectory_flag:
                        sin_period = panel.goal_period.val # period in seconds
                        x = (2*np.pi)/sin_period
                        theta_goal = panel.goal_amplitude.val*np.sin(x*current_time)
                    else:
                        theta_goal = panel.s_theta_goal.val
                    vel_goal = np.deg2rad(panel.s_vel_goal.val)
                    coact_goal = panel.s_coact_goal.val

                if use_pid:
                    action = pid_controller.step(obs[0], obs[1], theta_goal, coact_goal)
                    if panel.impulsed_triggered_flag:
                        panel.impulsed_triggered_flag = False
                elif use_model:
                    if hold_step_count >= 10:
                        e_theta = _wrap_pi(obs[0] - theta_goal)
                        e_vel = obs[1] - vel_goal
                        u_c = (action[0] + action[1]) / env_params.r1
                        e_uc = u_c - coact_goal
                        obs_appended = np.concatenate((obs, np.array([np.cos(theta_goal), np.sin(theta_goal), e_theta])))
                        
                        obs_norm = normalize_obs(obs_appended)

                        action, _ = model.predict(obs_norm, deterministic=True) 
                        action = _map_reparam_to_torques(action, coact_goal, env_params.r1)
                        hold_step_count = 0
                    if hold_step_count == 1:
                        if panel.impulsed_triggered_flag:
                            panel.impulsed_triggered_flag = False
                else:
                    action = panel.actions.as_array()
                    if panel.impulsed_triggered_flag:
                        panel.impulsed_triggered_flag = False
                obs, info = env.step(action)
                current_time = info.get("t")
                acc -= dt
            # Render at a target FPS (wall-clock based)
            if now - last_render >= render_dt:
                # You can pass the sim time from env if you track it, or compute it
                renderer.update(obs, action, np.array([theta_goal, coact_goal]), t=current_time)
                last_render = now
    except KeyboardInterrupt:
        pass
    finally:
        end_time = time.perf_counter()
        renderer.close()
        print(f"number of cycles: {count}, elapsed time: {end_time - start_time:.3f}")

if __name__ == "__main__":
    main()
