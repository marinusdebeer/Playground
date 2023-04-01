import gymnasium as gym
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation = env.reset()
action = 0
done = False

while not done or True:
    action = env.action_space.sample()
    
    observation, reward, done, _, info = env.step(action)
    print(observation.shape)
    
env.close()