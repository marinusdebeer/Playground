# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import random
import operator
class actor(tf.keras.Model):
    def __init__(self):
        super(actor, self).__init__()
        self.d1 = layers.Dense(256, activation="relu")
        self.d2 = layers.Dense(256, activation="relu")
        self.d3 = layers.Dense(num_actions, activation="tanh")

    @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)*upper_bound
    
class critic(tf.keras.Model):
    def __init__(self):
        super(critic, self).__init__()
        self.d1 = layers.Dense(16, activation="relu")
        self.d2 = layers.Dense(32, activation="relu")
        self.d3 = layers.Dense(256, activation="relu")
        self.d4 = layers.Dense(256, activation="relu")
        self.d5 = layers.Dense(1)

    @tf.function
    def call(self, val):
        x, a = val
        x = self.d1(x)
        x = self.d2(x)
        x = tf.concat([x, a], axis=1)
        x = self.d3(x)
        x = self.d4(x)
        return self.d5(x)

""" class Buffer:
    def __init__(self, buffer_capacity=100_000, batch_size=64):
        self.memory = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    @tf.function
    def update(self, states, actions, rewards, next_states,):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_states, training=True)
            y = rewards + gamma * target_critic([next_states, target_actions], training=True)
            critic_value = critic_model([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            pred_actions = actor_model(states, training=True)
            critic_value = critic_model([states, pred_actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    def learn(self):
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        # print(len(samples))
        states, actions, rewards, next_states = zip(*samples)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(states)
        action_batch = tf.convert_to_tensor(actions)
        reward_batch = tf.convert_to_tensor(rewards)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_states)

        # self.update(states, actions, rewards, next_states)
        self.update(state_batch, action_batch, reward_batch, next_state_batch) """



class Buffer:
    def __init__(self, buffer_capacity=100_000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_capacity)

        # Its tells us num of times remember() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def remember(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1
        self.memory.append(obs_tuple)

    @tf.function
    def update(self, states, actions, rewards, next_states,):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_states, training=True)
            y = rewards + gamma * target_critic([next_states, target_actions], training=True)
            critic_value = critic_model([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            pred_actions = actor_model(states, training=True)
            critic_value = critic_model([states, pred_actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def act(state):
    if np.random.random() < epsilon:
        return np.random.uniform(low=lower_bound, high=upper_bound, size=num_actions)
    return actor_model(state).numpy()[0]

problem = "Pusher-v4"
# problem = "Pendulum-v1"
# problem = "Humanoid-v4"
env = gym.make(problem)
# env = gym.make(problem, render_mode="human")
if not os.path.exists(problem):
    os.makedirs(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

total_episodes = 10_000
epsilon = 1.0
initial_epsilon = 1.0
decay = 0.04
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.5 # 0 means no update 1 means full update
steps_per_episode = 300
steps = 0
update_target_every = 500
train_every = 50
batch_size = 2048
buffer_size = 100_000
buffer = Buffer(buffer_size, batch_size)
pretrained = False
# actor_model = get_actor()
actor_model = actor()
critic_model = critic()

# target_actor = get_actor()
target_actor = actor()
target_critic = critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.001
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


if pretrained:
    state = env.reset()
    state = state[0]
    state = np.array(state)
    state = state[np.newaxis, :]
    action = actor_model(state).numpy()[0]
    action = target_actor(state).numpy()[0]
    # print("\n")
    # print((1, state, action))
    # print("\n")
    # critic_model((state, action))
    # action = target_actor(state)
    # target_critic((state, action))
    # print(state)

    actor_model.load_weights(f'{problem}/actor_100.h5')
    # critic_model.load_weights(f'{problem}/critic_100.h5')
    target_actor.load_weights(f'{problem}/target_actor_100.h5')
    # target_critic.load_weights(f'{problem}/target_critic_100.h5')

print("Size of State Space ->  {}".format(num_states))
print("Size of Action Space ->  {}".format(num_actions))
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
start = time.time()
# Takes about 4 min to train
for ep in range(1, total_episodes+1):
    ep_start = time.time()
    state = env.reset()
    state = state[0]
    episodic_reward = 0
    learn_time = 0
    action_time = 0
    while True:
        tf_state = np.array(state)
        tf_state = tf_state[np.newaxis, :]
        action_start = time.time()
        action = act(tf_state)
        action_end = time.time()
        action_time += action_end - action_start
        next_state, reward, done, _, info = env.step(action)
        buffer.remember((state, action, reward, next_state))
        # buffer.remember(state, action, reward, next_state)
        episodic_reward += reward
        if steps % train_every == 0:
            learn_start = time.time()
            buffer.learn()
            learn_end = time.time()
            learn_time += learn_end - learn_start
        if steps % update_target_every == 0:
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

        state = next_state
        steps += 1

        # End this episode when `done` is True
        if done or steps % steps_per_episode == 0:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = round(np.mean(ep_reward_list[-10:]))
    ep_time = time.time() - ep_start
    epsilon = initial_epsilon * (decay ** (ep/total_episodes))
    if ep % 10 == 0:
        actor_model.save_weights(f"{problem}/actor_{ep}.h5")
        critic_model.save_weights(f"{problem}/critic_{ep}.h5")

        target_actor.save_weights(f"{problem}/target_actor_{ep}.h5")
        target_critic.save_weights(f"{problem}/target_critic_{ep}.h5")
        avg_reward_list.append(avg_reward)
        print(f"Episode: {ep}, avg10: {round(np.mean(ep_reward_list[-10:]))}, avg50: {round(np.mean(ep_reward_list[-50:]))}, avg100: {round(np.mean(ep_reward_list[-100:]))}, avg_reward: {round(np.mean(ep_reward_list))}, epsilon: {round(epsilon, 4)}")
end = time.time()
total_time = end - start
avg_reward = round(np.mean(ep_reward_list))
avg_time = round((total_time)/total_episodes, 2)
training_score = avg_reward*avg_time
print("\n------------------------------------------------------------------------")
print(f"Total episodes: {total_episodes}, gamma: {gamma}, tau: {tau}, buffer_size: {buffer_size}, batch_size: {batch_size}, critic_lr: {critic_lr}, actor_lr: {actor_lr}, decay: {decay}, update_target_every: {update_target_every}, train_every: {train_every}, steps_per_episode: {steps_per_episode}")
# print(f"avg10_reward: {np.mean(ep_reward_list[-10:])}, avg50_reward: {np.mean(ep_reward_list[-50:])}")
print(f"Total time: {round(total_time, 2)}s, avg_time: {avg_time}, avg reward: {round(np.mean(ep_reward_list))}, training score: {training_score}")
print("------------------------------------------------------------------------\n")
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()