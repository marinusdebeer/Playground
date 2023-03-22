import carla
import tensorflow as tf
import numpy as np
import glob
import os
import sys
import cv2
import time
import random
from tqdm import tqdm
from keras.optimizers import Adam
from threading import Thread
from collections import deque
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 30
MIN_REPLAY_MEMORY_SIZE = 20
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
AGGREGATE_STATS_EVERY = 10
MODEL_NAME = "Xception"
DISCOUNT=0.993
EPISODES = 1000
EPSILON_DECAY = 0.993
epsilon = 0.993

class CarEnv:
    def __init__(self):
        # Initialize the Carla environment and agent
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.prev_speed = None
        self.prev_steering = None
        self.front_camera = None
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        self.vehicle_spawn_point = carla.Transform(carla.Location(x=-25, y=135, z=0.1), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))
        
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.camera_bp.set_attribute("fov", f"110")
        self.camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    # Define the reward function
    def get_reward(self, prev_position, position, speed):
        # Calculate the distance traveled
        distance = np.linalg.norm(np.array(position) - np.array(prev_position))
        # Calculate the reward based on distance traveled and speed
        reward = distance * speed / 100
        return reward
    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.prev_position = (-25, 135)
        self.front_camera = None
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.vehicle_spawn_point)
        self.actor_list.append(self.vehicle)
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        while self.front_camera is None:
            time.sleep(0.01)

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        return self.front_camera
    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        gray = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
        if True:
            cv2.imshow("", gray)
            cv2.waitKey(1)
        image_array = np.expand_dims(i3, axis=0)
        image_array = image_array / 255.
        self.front_camera = image_array

    def step(self, action):
        done = False
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=action))

        location = self.vehicle.get_location()
        position = (location.x, location.y)
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm(np.array([velocity.x, velocity.y]))
        
        reward = self.get_reward(self.prev_position, position, speed)
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        
        # Update the previous position and speed
        self.prev_position = position
        self.prev_speed = speed
        # print("reward: ", reward)
        # Return the reward
        return self.front_camera, reward, done

# Define the reinforcement learning agent
class RLAgent:
    def __init__(self):
        self.model = self.createModel()
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())
        self.terminate = False
        self.training_initialized = False
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        
    def createModel(self):
        # Define the deep learning model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        # print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        print('STARTING')
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # minibatch = np.array(random.sample(self.replay_memory, MINIBATCH_SIZE))
        current_states = np.array([transition[0] for transition in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        new_current_states = np.array([transition[3] for transition in minibatch])
        done_flags = np.array([i[4] for i in minibatch])
        print('training',current_states[0].shape)
        discrete_actions = (((actions + 1) / 2) * 100).astype(int)
        for i in range(MINIBATCH_SIZE):
            current_q_values = self.model.predict([current_states[i]])
            discrete_current_q_values = (((current_q_values + 1) / 2) * 100).astype(int)
            print("discrete_current_q_values",discrete_current_q_values)
            next_q_values = self.model.predict([new_current_states[i]]).max(axis=1)
            discrete_next_q_values = (((next_q_values + 1) / 2) * 100).astype(int)
            discrete_current_q_values[0, discrete_actions[i]] = rewards[i] + (1 - done_flags[i]) * DISCOUNT * discrete_next_q_values
            self.model.train_on_batch(np.array([current_states[i]]), discrete_current_q_values)

        # current_q_values = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)
        # next_q_values = self.model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE).max(axis=1)
        # current_q_values[np.arange(MINIBATCH_SIZE), actions] = rewards + (1 - done_flags) * DISCOUNT * next_q_values
        # self.model.train_on_batch(current_states, current_q_values)

        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        # print(new_current_states)
    def train_in_loop(self):
        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

    def get_action(self, observation):
        # Use the deep learning model to predict the steering command
        steering = self.model.predict(observation, verbose=0)
        # Add some noise to the steering command to encourage exploration
        steering += np.random.normal(scale=0.1)

        # Limit the steering command to the range [-1, 1]
        steering = float(np.clip(steering, -1, 1)[0][0])

        # Update the previous steering command
        self.prev_steering = steering

        return steering
    
    



if __name__ == '__main__':
    env = CarEnv()
    agent = RLAgent()
    initial = np.ones((IM_HEIGHT, IM_WIDTH, 3))
    initial = np.array(initial).reshape(-1, *initial.shape)/255
    agent.get_action(initial)
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    ep_rewards = []

    # Run the simulation loop
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        current_state = env.reset()
        episode_reward = 0
        done = False
        episode_start = time.time()
        while True:
            # print(current_state)
            # Get the action from the agent
            action = agent.get_action(current_state)
            # print("action", action)

            # Update the agent with the current state and get the reward
            new_state, reward, done = env.step(action)
            episode_reward += reward
            # Every step we update replay memory
            # print('main',current_state.shape)
            
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            if done:
                break

         # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()
        ep_rewards.append(episode_reward)
        epsilon = pow(EPSILON_DECAY, episode)
        if not episode % 10:
            agent.model.save(f"model_ep_{episode}.h5")
     # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f"model_ep_{episode}.h5")