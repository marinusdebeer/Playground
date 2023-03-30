import gymnasium as gym
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation = env.reset()
action = 0
done = False
# Initialize curses
# screen = curses.initscr()
# curses.cbreak()
# curses.noecho()
# screen.keypad(True)
# screen.nodelay(True)
while not done or True:
    # env.render()
     # Read user input
    # key = screen.getch()
    # print(key)
    # Select action based on input
    # if key == curses.KEY_RIGHT:
        # action = 2
    # elif key == curses.KEY_LEFT:
        # action = 3
    action = env.action_space.sample()
    
    observation, reward, done, _, info = env.step(action)
    print(observation.shape)
    
env.close()

# Clean up curses
# curses.nocbreak()
# screen.keypad(False)
# curses.echo()
# curses.endwin()