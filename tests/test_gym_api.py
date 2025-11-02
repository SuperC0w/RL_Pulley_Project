import numpy as np
from env.gym_env import PulleyEnvGym

def test_step_signature_and_dtype():
    env = PulleyEnvGym()
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    for _ in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        if term or trunc:
            break

def test_determinism_with_seed():
    env1 = PulleyEnvGym()
    env2 = PulleyEnvGym()
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    assert np.allclose(o1, o2)

def test_time_limit_truncation():
    env = PulleyEnvGym(max_steps=5)
    env.reset(seed=0)
    done = False
    for _ in range(10):
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        if term or trunc:
            done = True
            break
    assert done
