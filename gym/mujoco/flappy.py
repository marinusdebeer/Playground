# https://github.com/markub3327/flappy-bird-gymnasium
# Use conda env: gym_env_py_39

import time
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0")


while True:
    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing:
        obs, reward, terminated, _, info = env.step(action)
        
        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS
        
        # Checking if the player is still alive
        if terminated:
            break

env.close()