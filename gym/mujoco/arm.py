import gymnasium as gym
# env = gym.make('Pusher-v4', render_mode='human')
env = gym.make('Pusher-v4', render_mode='human')
env.reset()
while True:
    env.render()
    env.step(env.action_space.sample()) # take a random action