import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Create the Atari Breakout environment
env = gym.make("BreakoutNoFrameskip-v4")

# Apply Atari preprocessing
env = AtariWrapper(env)

# Stack frames (to include temporal information)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Initialize the DQN agent
model = DQN("CnnPolicy", env, verbose=1, buffer_size=100000, learning_starts=10000)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("dqn_breakout")

# Test the trained model
env.reset()
done = False
while not done:
    action, _states = model.predict(env, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
