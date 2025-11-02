from .gym_env import PulleyEnvGym  # re-export

from gymnasium.envs.registration import register

register(
    id="PulleyEnv-v0",
    entry_point="env:PulleyEnvGym",   # module:class
)