import gymnasium as gym
import time
import cv2
import numpy as np

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation = env.reset()
action = 1
fire = True
done = False
move_left = True
start_time = time.time()
lives = 5

# print(observation)
def preprocess_observation(image):
    # Extract the image and metadata from the observation tuple
    # image, metadata = observation
    # Resize the image to a smaller size
    image = cv2.resize(image, (84, 84))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

num_states = 84 #observation.shape[1]
num_actions = 4 #env.action_space.n
q_table = np.random.rand(num_states, num_actions)
print(q_table.shape)
# print(q_table)
while not done:
  observation, reward, done, _, info = env.step(1)  # Call with dummy action to get initial info
  # Check if one second has passed
  elapsed_time = time.time() - start_time
  if elapsed_time >= 1:
    move_left = not move_left
    start_time = time.time()
  # Move left or right based on the value of move_left
  if fire:
    action = 1
  elif move_left:
    action = 3  # Move left
  else:
    action = 2  # Move right
  observation, reward, done, _, info = env.step(action)
  observation = preprocess_observation(observation)
  if(info["lives"] < lives):
    lives = info["lives"]
    fire = True
    print(observation.shape)
  else:
    fire = False

# Close the environment
env.close()
