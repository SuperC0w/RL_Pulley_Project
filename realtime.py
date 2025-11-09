import time
import numpy as np
import matplotlib.pyplot as plt

from env.params import PulleyParams
from env.pulley_env import PulleyEnv
from ui.render_matplotlib import MatplotlibRenderer
from ui.controls import ControlPanel
from controllers.pid import PID
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.gym_env import PulleyEnvGym

from stable_baselines3 import SAC

DT_SIM = 0.001
FPS = 90

USE_PID = False
USE_MODEL = True

def _wrap_pi(x):
        """
        Wrap to (-pi, pi]
        """
        return (x + np.pi) % (2*np.pi) - np.pi

def main():
    dt = DT_SIM
    RENDER_DT = 1.0 / FPS
    env_params = PulleyParams()
    env = PulleyEnv(env_params, dt=dt, max_steps=100_000, seed=0)
    action = np.zeros(2, dtype=np.float32)

    renderer = MatplotlibRenderer(env_params, dt_sim=env.dt)
    renderer.draw_static()

    if USE_PID and USE_MODEL:
        raise ValueError("Both USE_PID and USE_MODEL cannot be true at the same time")

    if USE_PID:
        pid_controller = PID()
    elif USE_MODEL:
        # model = SAC.load("./models/SAC/best/best_model.zip", device="cpu")
        # vecnorm_path = "./models/SAC/best/best_vecnorm.pkl"
        # model = SAC.load("./models/SAC/best/current_best.zip", device="cpu")
        # vecnorm_path = "./models/SAC/best/current_best.pkl"
        model = SAC.load("./models/SAC/best/test2.zip", device="cpu")
        vecnorm_path = "./models/SAC/best/test2.pkl"

        venv_for_norm = DummyVecEnv([lambda: PulleyEnvGym(dt=DT_SIM, max_steps=100_000, seed=0)])
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
        """Reset env state and renderer buffers (called by the Reset button)."""
        obs, *_ = env.reset()
        renderer.clear_buffers()
        # Optionally show the reset frame immediately:
        # renderer.update(obs, np.array([0, 0]), t=0.0)
    
    # Open the separate controls window (non-blocking)
    panel = ControlPanel(env_params, tau1_lim=2.0e-1, tau2_lim=2.0e-1, F_range=(-2.0, 2.0),
                         show_param_sliders=True, on_reset=do_reset)
    
    plt.ion()                 # turn on interactive mode
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
    acc  = 0.0                      # accumulator of real time
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
                acc = MAX_ACC  # drop sim frames if we fell behind badly

            # Step the simulation in fixed increments (could be 0, 1, or many steps)
            while acc >= dt:
                count += 1
                hold_step_count += 1
                # get goal value from control panel
                if USE_PID or USE_MODEL:
                    if panel.follow_trajectory_flag:
                        sin_period = panel.goal_period.val # period in seconds
                        x = (2*np.pi)/sin_period
                        theta_goal = panel.goal_amplitude.val*np.sin(x*current_time)
                    else:
                        theta_goal = panel.s_theta_goal.val
                    
                    coact_goal = panel.coact_goal.val

                if USE_PID:
                    action = pid_controller.step(obs[0], obs[1], theta_goal, coact_goal)
                    if panel.impulsed_triggered_flag == True:
                        env.trigger_impulse()
                        panel.impulsed_triggered_flag = False
                elif USE_MODEL:
                    if hold_step_count >= 10:
                        e_theta = _wrap_pi(obs[0] - theta_goal)
                        u_c = (action[0] + action[1]) / env_params.r1
                        e_uc = u_c - coact_goal
                        obs_appended = np.concatenate((obs, np.array([np.cos(theta_goal), np.sin(theta_goal), e_theta, 
                                                                    coact_goal, e_uc])))
                        
                        obs_norm = normalize_obs(obs_appended)

                        action, _ = model.predict(obs_norm, deterministic=True) 
                        hold_step_count = 0
                    if hold_step_count == 1:
                        if panel.impulsed_triggered_flag == True:
                            env.trigger_impulse()
                            panel.impulsed_triggered_flag = False
                else:
                    action = panel.actions.as_array()
                # print(action)
                obs, info = env.step(action)
                # print(obs[1])
                current_time = info.get("t")
                acc -= dt
            # Render at a target FPS (wall-clock based)
            if now - last_render >= RENDER_DT:
                # You can pass the sim time from env if you track it, or compute it
                renderer.update(obs, action, np.array([theta_goal, coact_goal]), t=current_time)
                last_render = now

            # time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        end_time = time.perf_counter()
        renderer.close()
        print(f"number of cycles: {count}, elapsed time: {end_time - start_time:.3f}")

if __name__ == "__main__":
    main()
