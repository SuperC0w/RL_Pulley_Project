from .gym_env import PulleyEnvGym  # re-export
from .gym_env_tau import PulleyEnvGym as PulleyEnvGymTau

from gymnasium.envs.registration import register

register(
    id="PulleyEnv-v0",
    entry_point="env:PulleyEnvGym",   # module:class
)

register(
    id="PulleyEnv-Tau",
    entry_point="env:PulleyEnvGymTau"
)