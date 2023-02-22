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

# tf.compat.v1.disable_eager_execution()

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 30
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "pos_vel"
# MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.993 #0.9985
MIN_EPSILON = 0.1

AGGREGATE_STATS_EVERY = 10
TRIAL=14
MODEL_PATH = 'models/pos_vel_-1040_-1243_-781_ep_30_trail_12.model'
use_prev_model = False

def save(average_reward, min_reward, max_reward, episode):
    agent.model.save(f'models/{MODEL_NAME}_{round(average_reward)}_{round(min_reward)}_{round(max_reward)}_ep_{episode}_trail_{TRIAL}.model')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        # self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()
        self.step += 1


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    goal_position = carla.Location(x=-109, y=28.8, z=0)
    # starting position = (x=-11.1, y=138.5, z=0)


    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.spawn_point = carla.Transform(carla.Location(x=-11.1, y=138.5, z=0.1), carla.Rotation(yaw=180))

    # Define a function that computes the distance to the goal
    
    def distance_to_goal(self, location):
        return location.distance(self.goal_position)

    # Define the reward function
    def goal_based_reward(self, location):
        # Compute the distance to the goal
        max_dist = 147
        dist = self.distance_to_goal(location)
        return max_dist - round(dist)
        # print("dist: ", dist)
        # Set the maximum distance to the goal
        # max_dist = 148
        # # Compute the reward as a function of the distance
        # if dist < max_dist:
        #     reward = 1 - dist / max_dist
        # else:
        #     reward = -1
        # return round(reward, 3)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        self.actor_list.append(self.vehicle)
        self.vehicle_prev_loc = { 'x': -11.1, 'y': 138.5}

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # Convert state dictionary to numpy array
        tem = {'lx':-11.1, 'ly':138.5, 'vx':0, 'vy':0}
        # Convert state dictionary to numpy array
        tem = np.array([tem['lx'], tem['ly'], tem['vx'], tem['vy']])
        # Reshape the state array and normalize the values
        tem = tem.reshape(-1, 4) / 255
        return tem
        # return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    
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

        # Get the vehicle's location and velocity
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
        done = False
        
        moved = round(abs(loc.x - self.vehicle_prev_loc['x']) + abs(loc.y - self.vehicle_prev_loc['y']), 2)
        
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            reward = round(pow(kmh, 1.5) / 20)
            if kmh < 20:
                reward = -1
            distance = round(abs(loc.x - (-11.1)) + abs(loc.y - 138.5))
            distance_per_time = distance / (time.time() - self.episode_start)
            reward = distance_per_time
            goal = self.goal_based_reward(loc)
            # print("goal: ", goal)
            reward = goal
            # print(reward)
            # print("distance:", distance)
            # print("moved:", moved)
        
        # if kmh > 60:  
            # print("kmh: ", kmh)
        self.vehicle_prev_loc = {'x': loc.x, 'y': loc.y}
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        # Convert state dictionary to numpy array
        observation = np.array([loc.x, loc.y, vel.x, vel.y])
        # Reshape the state array and normalize the values
        observation = observation.reshape(-1, 4) / 255


        # Return the observation, reward, done flag, and an empty info dictionary
        return observation, reward, done, None


class DQNAgent:
    def __init__(self):
        if use_prev_model:
            self.model = load_model(MODEL_PATH)
            self.target_model = load_model(MODEL_PATH)
        else:    
            self.model = self.create_model()
            self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        input_state = Input(shape=(4,), name='state')
        x = Dense(32, activation='relu')(input_state)
        predictions = Dense(4, activation="linear")(x)
        model = Model(inputs=input_state, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print('minibatch: ', minibatch)
        current_states = np.array([transition[0] for transition in minibatch])
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        # # Convert state dictionary to numpy array
        # state_array = np.array([state['lx'], state['ly'], state['vx'], state['vy']])

        # # Reshape the state array and normalize the values
        # state_array = state_array.reshape(-1, 4) / 255
        

        # Use the model to predict the Q-values for the current state
        q_values = self.model.predict(state)[0]
        # print("state: ", state, " state_array: ", state_array, " q_values: ", q_values)

        return q_values

    def train_in_loop(self):
        X = np.random.uniform(size=(1, 4)).astype(np.float32)
        y = np.random.uniform(size=(1, 4)).astype(np.float32)
        self.model.fit(X, y, verbose=0, batch_size=1)
        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60
    np.set_printoptions(precision=4)
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    # random.seed(1)
    np.random.seed(1)
    # Set the random seed
    tfrandom.set_seed(1)
    # tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create a TensorFlow configuration object
    config = tf.compat.v1.ConfigProto()

    # Set the GPU options
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION

    # Create a TensorFlow session with the specified configuration
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # Concatenate the location and velocity into a single observation
    tem = {'lx':-11.1, 'ly':138.5, 'vx':0, 'vy':0}
    # Convert state dictionary to numpy array
    tem = np.array([tem['lx'], tem['ly'], tem['vx'], tem['vy']])
    # Reshape the state array and normalize the values
    tem = tem.reshape(-1, 4) / 255
    agent.get_qs(tem)
    # agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:

            env.collision_hist = []

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:
                decs = None
                q = None
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    # print("QUERY: ", current_state)
                    q = agent.get_qs(current_state)
                    decs = 'chosen'
                    action = np.argmax(q)
                    print('action: ', q, action)
                    
                else:
                    decs = 'rand'
                    # Get random action
                    action_probs = [0.25, 0.4, 0.25, 0.1]  # Probabilities for each action
                    action = np.random.choice(len(action_probs), p=action_probs)
                    # action = np.random.randint(0, 4)
                    # print("RANDOM: ", action)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                # print('action: ', decs, q, action)
                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
                time.sleep(0.3)
                if done:
                    print('Episode reward: ', episode_reward)
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                print("episode: ", episode)
                print("average_reward: ", average_reward, " min_reward: ", min_reward, " max_reward: ", max_reward)
                print("EPSILON: ", epsilon)
                # Save model, but only when min reward is greater or equal a set value
                # if average_reward >= MIN_REWARD or not episode % (AGGREGATE_STATS_EVERY*5) or episode == 1:

                # agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                # agent.model.save(f'models/{MODEL_NAME}_{average_reward}_{min_reward}_{max_reward}.{episode}_trial_{TRIAL}.model')
                save(average_reward, min_reward, max_reward, episode)
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                # epsilon *= EPSILON_DECAY
                epsilon = pow(EPSILON_DECAY, episode)
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    # agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    # agent.model.save(f'models/{MODEL_NAME}_{round(average_reward, 0)}_{round(min_reward, 0)}_{round(max_reward, 0)}.{episode}_trail_{TRIAL}.model')
    save(average_reward, min_reward, max_reward, episode)