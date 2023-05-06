import gymnasium as gym
# from gymnasium.envs import ale
# rom_path = 'AutoROM/roms/breakout.bin' 
# ale.ale_import('AutoROM/roms/breakout.bin' )
# env = gym.make('ALE/Breakout-v5')
env = gym.make('Breakout-v4', render_mode='human')
env.reset()

for _ in range(1000):
   env.step(env.action_space.sample())

env.close()
