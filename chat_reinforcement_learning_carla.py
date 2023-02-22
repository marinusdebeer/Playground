import carla
import numpy as np

import glob
import os
import sys
import random
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
client = carla.Client('localhost', 2000)
client.set_timeout(3.0)

FOLDER="chat_q_tables"
# Define the environment
class CarlaEnv:
    def __init__(self):
        self.client = client
        # self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.audi.etron')[0]
        self.vehicle = None
        self.sensor_actor = None

    def reset(self):
        print('reseting...')
        if self.vehicle:
            self.vehicle.destroy()
        if self.sensor_actor:
            self.sensor_actor.destroy()
        self.actor_list = []
        self.collision_hist = []
        # spawn_point = random.choice(self.world.get_map().get_spawn_points())
        spawn_point = carla.Transform(carla.Location(x=-11.1, y=138.5, z=0.1), carla.Rotation(yaw=180))
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        self.sensor = self.blueprint_library.find('sensor.other.collision')
        self.sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0))
        self.sensor_actor = self.world.spawn_actor(self.sensor, self.sensor_transform, attach_to=self.vehicle)
        self.sensor_actor.listen(lambda event: self.collision_hist.append(event))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.world.tick()
        return self._get_observation()

    def _get_observation(self):
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        collision = len(self.collision_hist)
        return np.array([vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.rotation.yaw, velocity.x, velocity.y, velocity.z, collision])

    def step(self, action):
        if action == 0:
            control = carla.VehicleControl(throttle=1, steer=-0.5)
        elif action == 1:
            control = carla.VehicleControl(throttle=1, steer=0.5)
        elif action == 2:
            control = carla.VehicleControl(throttle=1, steer=0.0)
        self.vehicle.apply_control(control)
        self.world.tick()
        reward = self._get_reward()
        done = self._is_done()
        return self._get_observation(), reward, done

    def _get_reward(self):
        collision = len(self.collision_hist)
        if collision > 0:
            return -100
        else:
            velocity = self.vehicle.get_velocity()
            reward = velocity.x
            return reward

    def _is_done(self):
        collision = len(self.collision_hist)
        return collision > 0

# Define the Q-learning algorithm
class QLearning:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def discretize_state(state):
        if(type(state) == tuple):
            state = np.array([state[0][0], state[0][1]])
        discrete_state = (state - env.observation_space.low) * np.array([10, 100])
        return np.round(discrete_state, 0).astype(int)
    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.q_table.shape[1])
        else:
            action = np.argmax(self.q_table[state.argmax()])
        return action

    def update(self, state, action, reward, next_state, done):
        # print(type(next_state))
        # q_next = np.max(self.q_table[next_state])
        q_next = np.max(self.q_table[next_state.argmax()])
        td_target = reward + self.discount_factor * q_next * (1 - done)
        td_error = td_target - self.q_table[state.argmax(), action]
        self.q_table[state.argmax(), action] += self.learning_rate * td_error

# Train the agent
env = CarlaEnv()
num_states = 7
num_actions = 3
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
num_episodes = 11
SAVE_EVERY=10

q_learning = QLearning(num_states, num_actions, learning_rate, discount_factor)
reward_history = []


for episode in range(num_episodes):
    state = env.reset()
    terminated = False
    total_reward = 0
    while not terminated:
        action = q_learning.get_action(state, epsilon)
        next_state, reward, terminated = env.step(action)
        q_learning.update(state, action, reward, next_state, terminated)
        state = next_state
        total_reward += reward
        print("action: ", action, " reward: ", reward)
    reward_history.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1} completed with total reward: {total_reward}")
    if not episode % SAVE_EVERY:
        np.save(f"{FOLDER}/{episode}-qtable.npy", q_learning.q_table)
# except:
    # print('exception')
# finally:
    # Destroy an actor at end of episode
# actors = client.get_world().get_actors()
# for actor in actors:
    # actor.destroy()
# Test the agent
state = env.reset()
terminated = False
while not terminated:
    action = np.argmax(q_learning.q_table[state.argmax()])
    next_state, reward, terminated = env.step(action)
    state = next_state

