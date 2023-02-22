import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from tensorflow import random as tfrandom
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorflow.keras.models import load_model
from keras.layers import Input


import tensorflow as tf
from keras import backend as backend
from threading import Thread

from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
# Q-learning parameters
MODEL_NAME = "pos_vel"
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
EPISODES = 1000
STATS_EVERY = 10
FOLDER = 'qtables'
SECONDS_PER_EPISODE = 10
AGGREGATE_STATS_EVERY = 10
TRIAL=12

def save(average_reward, min_reward, max_reward, episode):
    agent.model.save(f'models/{MODEL_NAME}_{round(average_reward)}_{round(min_reward)}_{round(max_reward)}_ep_{episode}_trail_{TRIAL}.model')
class CarEnv:
    STEER_AMT = 1.0
    front_camera = None


    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.spawn_point = carla.Transform(carla.Location(x=-11.1, y=138.5, z=0.1), carla.Rotation(yaw=180))

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        self.actor_list.append(self.vehicle)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))


        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return {'lx':-11.1, 'ly':138.5, 'vx':0, 'vy':0}
        # return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        # left
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-1*self.STEER_AMT, brake=0.0))
        # straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, brake=0.0))
        # right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=1*self.STEER_AMT, brake=0.0))
        # brake
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake=1.0))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            reward = (kmh / 20) - 5  

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        # Get the vehicle's location and velocity
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        observation ={ 'lx': loc.x, 'ly': loc.y, 'vx': vel.x, 'vy': vel.y}
        return observation, reward, done, None

env = CarEnv()
# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Initialize the Q-table with zeros
q_table = np.zeros([4,4])


# Define the action function
def get_action(state):
    if np.random.uniform() < epsilon:
        # Select a random action
        action_probs = [0.2, 0.5, 0.2, 0.1]  # Probabilities for each action
        action = np.random.choice(len(action_probs), p=action_probs)
    else:
        # Select the action with the highest Q-value
        action = np.argmax(q_table[state])
    return action

def get_discrete_state(state):
    # Scale state values to be between 0 and 1
    lx = state['lx'] / 255
    ly = state['ly'] / 255
    vx = state['vx'] / 255
    vy = state['vy'] / 255
    
    # Discretize state values into 10 equally-sized bins
    num_bins = 10
    lx_bin = int(lx * num_bins)
    ly_bin = int(ly * num_bins)
    vx_bin = int(vx * num_bins)
    vy_bin = int(vy * num_bins)
    
    return np.array([lx_bin, ly_bin, vx_bin, vy_bin])


def update_q_table(state, action, reward, next_state):
    # Convert states to discrete states
    state_discrete = get_discrete_state(state)
    next_state_discrete = get_discrete_state(next_state)
    
    print("state_discrete", state_discrete)
    print("next_state_discrete", next_state_discrete)

    # Update Q-value for the current state-action pair
    temp = q_table[state_discrete[0], state_discrete[1], state_discrete[2], state_discrete[3]]
    print(temp)
    old_q_value = q_table[state_discrete[0], state_discrete[1], state_discrete[2], state_discrete[3]][action]
    max_q_value = np.max(q_table[next_state_discrete])
    q_table[state_discrete][action] = (1 - alpha) * old_q_value + alpha * (reward + gamma * max_q_value)

    return q_table

""" def discretize_state(state):
if(type(state) == tuple):
    state = np.array([state[0][0], state[0][1]])
discrete_state = (state - env.observation_space.low) * np.array([10, 100])
return np.round(discrete_state, 0).astype(int) """

# Update the Q-table based on the observed reward
""" def update_q_table(state, action, reward, next_state):
    state_array = np.array([state['lx'], state['ly'], state['vx'], state['vy']])
    state_array = state_array.reshape(-1, 4) / 255
    print("state_array", state_array)

    next_state_array = np.array([next_state['lx'], next_state['ly'], next_state['vx'], next_state['vy']])
    next_state_array = next_state_array.reshape(-1, 4) / 255
    print("next_state_array", next_state_array)
    print("next_state_array", next_state_array[0][1])
    # print("q_table", q_table)
    max_q_value = np.max(q_table[next_state_array[0][0], next_state_array[0][1], next_state_array[0][2], next_state_array[0][3]], action)
    old_q_value = q_table[state_array[0][0], state_array[0][1], state_array[0][2], state_array[0][3], action]
    q_table[state_array[0][0], state_array[0][1], state_array[0][2], state_array[0][3]][action] = (1 - alpha) * old_q_value + alpha * (reward + gamma * max_q_value) """

# Q-learning algorithm
for episode in range(1, EPISODES):
    state = env.reset()
    terminated = False
    timesteps = 0

    while not terminated:
        # Select an action
        action = get_action(state)

        # Take action
        next_state, reward, terminated, _ = env.step(action)

        # Update Q-value
        update_q_table(state, action, reward, next_state)

        state = next_state
        timesteps += 1

    # Decay epsilon
    epsilon = pow(epsilon_decay, episode)