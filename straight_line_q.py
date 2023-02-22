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


SECONDS_PER_EPISODE = 30
# Define the function for discretizing a continuous state
def discretize_state(state):
    x_idx = int((state[0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * n_x)
    y_idx = int((state[1] - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * n_y)
    vx_idx = int((state[2] - vx_bounds[0]) / (vx_bounds[1] - vx_bounds[0]) * n_vx)
    vy_idx = int((state[3] - vy_bounds[0]) / (vy_bounds[1] - vy_bounds[0]) * n_vy)
    # print((x_idx, y_idx, vx_idx, vy_idx))
    return (x_idx, y_idx, vx_idx, vy_idx)
class CarEnv:
    STEER_AMT = 1.0
    front_camera = None
    goal_position = carla.Location(x=-76, y=133, z=0)
    vehicle_prev_loc = carla.Location(x=81, y=133, z=0.1)
    im_width = 640
    im_height = 480
    SHOW_CAM = False

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_transform = carla.Transform(carla.Location(x=-6.5, z=2.4))

        self.camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_bp.set_attribute('fov', '110')
        self.video_file = None
        self.video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_fps = 30
        self.video_size = (self.im_width, self.im_height)
        self.video_writer = None
        
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.starting_position = [81 , 133, 0.0, 0.1 ]
        self.spawn_point = carla.Transform(carla.Location(x=81, y=133, z=0.1), carla.Rotation(yaw=180))
    def approaching_goal(self, prev, curr):
        dist = round(self.distance_to_goal(prev) - self.distance_to_goal(curr))
        # print(f"prev: {prev} curr: {curr} dist: {dist}")
        return dist

    def video(self, episode):
        self.video_file = f"{VIDEO_LOCATION}episode_{episode}.mp4"
        self.video_writer = cv2.VideoWriter(self.video_file, self.video_codec, self.video_fps, self.video_size)

    # Define a function that computes the distance to the goal
    def distance_to_goal(self, location):
        return location.distance(self.goal_position)

    # Define the reward function
    def goal_based_reward(self, location):
        dist = self.distance_to_goal(location)
        return -round(dist/10)
    
    def capture_and_save_image(self, camera, episode):
        if self.front_camera is not None:
            # Convert the image to BGR format
            img = cv2.cvtColor(self.front_camera, cv2.COLOR_RGB2BGR)
            # Save the image as a file
            cv2.imwrite('episode_{}.png'.format(episode), img)
        # image = camera.convert(carla.ColorConverter.Raw)
        # image.save_to_disk('episode_{}.png'.format(episode))

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.video_writer is not None:
            # Convert the image to BGR format
            img = cv2.cvtColor(i3, cv2.COLOR_RGB2BGR)
            # Write the frame to the video file
            self.video_writer.write(img)
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3
    
    def reset(self):
        
        self.collision_hist = []
        self.actor_list = []
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        self.actor_list.append(self.vehicle)
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(4)

        #attach camera for recordings
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(0.2)
        self.vehicle_prev_loc = self.vehicle.get_location()
        return discretize_state(self.starting_position)
        # return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        time.sleep(0.1)
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
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
        done = False
        reward = self.approaching_goal(self.vehicle_prev_loc, loc)
        # print(reward)
        if len(self.collision_hist) != 0:
            done = True
            reward = -100

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        if loc.x < -90 or loc.x > 100 or loc.y < 130 or loc.y > 141:
            reward = -100

        if loc.x < -90:
            loc.x = -89
            done = True
        if loc.x > 100:
            loc.x = 99
            done = True
        if loc.y < 130:
            loc.y = 131
            done = True
        if loc.y > 141:
            loc.y = 140
            done = True
        # goal_position = carla.Location(x=-76, y=133, z=0)
        # if loc.x < -74 and loc.x > -78 and loc.y > 131 and loc.y < 135:
        if loc.x < -76 and loc.y > 125 and loc.y < 138:
            reward = 100
            done = True
            print("GOAL REACHED!!!")
        self.vehicle_prev_loc = carla.Location(x=loc.x, y=loc.y, z=0.1)
        observation = discretize_state([loc.x, loc.y, vel.x, vel.y])
        return observation, reward, done, None

env = CarEnv()

x_bounds = (-90, 100)
y_bounds = (120, 150)
vx_bounds = (-40, 40)
vy_bounds = (-40, 40)
# n_x * n_y * n_vx * n_vy * 4 * 8 bytes
n_x = 190
n_y = 30
n_vx = 40
n_vy = 40
rand = 0
choice = 0
#           0.99    0.993   0.997    0.9997      0.9985  0.9997     0.99985     0.99997
# 50        0.61    0.704                        0.928              0.993
# 100       0.366   0.495                        0.861              0.985
# 300       0.049   0.122   0.406
# 500               0.03    0.223                0.472              0.928
# 1000     0.007    0.0008  0.050                0.223              0.861
# 10_000                              0.050      0.223    0.050     0.223
# 20_000                                         0.050              0.050
# 100_000                                                           0           0.050
# Define the Q-learning parameters
alpha = 0.2           # learning rate, how much past experience is valued
gamma = 0.995         # discount factor, importance of future rewards
epsilon = 0.997       # exploration, how much exploration to do
EPSILON_DECAY=0.997
num_episodes = 1000
training = True
use_existing_model = True
Q_TABLES_PATH='qtables/straight_line_qtable.npy'
SAVE_EVERY = 10
SAVE_LOCATION="F:/Coding/carla_q_learning_qtables/"
VIDEO_LOCATION="F:/Coding/recordings/"
RECORD = False

# Define the function for selecting an action using epsilon-greedy policy
def select_action(state, q_table):
    if np.random.random() > epsilon or not training:
        global choice
        choice += 1
        return np.argmax(q_table[state])
    else:
        global rand
        rand+=1
        action_probs = [0.15, 0.64, 0.15, 0.06]  # Probabilities for each action
        return np.random.choice(len(action_probs), p=action_probs)
       
# Define the function to save the Q-table to a file
def save_q_table(q_table, filename):
    np.save(filename, q_table)
q_table = np.zeros((n_x, n_y, n_vx, n_vy, 4))
q_table[:, :, :, :, 1] = 1
file_reward = 0
session_start = time.time()
if not training or use_existing_model:
    q_table = np.load(Q_TABLES_PATH)
for episode in range(num_episodes+1):
    state = env.reset()
    if RECORD:
        env.video(episode)
    episode_reward = 0
    epsilon = pow(EPSILON_DECAY, episode)
    while True:
        action = select_action(state, q_table)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        if training:
            q_table[state + (action,)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state + (action,)])
        state = next_state
        if done:
            break
    print("rand", rand, " choice", choice, "epsilon", epsilon)
    rand = 0
    choice = 0

    if env.video_writer is not None and RECORD:
        env.video_writer.release()
        env.video_writer = None
        if not reward == 100:
            os.remove(env.video_file)

    for actor in env.actor_list:
        actor.destroy()
    file_reward += episode_reward
    print(f"Episode {episode}: Total reward = {episode_reward}")
    if episode % SAVE_EVERY == 0 and training:
        print(f"Episode {episode}, file: {file_reward}, elapsed time: {time.time() - session_start}")
        q_table_filename = f"{SAVE_LOCATION}q_table_episode{episode}_{file_reward}.npy"
        save_q_table(q_table, q_table_filename)
        file_reward = 0
        # env.capture_and_save_image(env.camera, episode)

session_end = time.time()
print("elapsed: ", session_end - session_start,"Finshed: ", session_end)