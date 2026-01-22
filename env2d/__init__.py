# will need to implemented later when the gym_env is created 
from .gym_env import PulleyEnvGym  # re-export

from gymnasium.envs.registration import register

register(
    id="PulleyEnv2D-v0",
    entry_point="env2d:PulleyEnvGym",   # module:class
)