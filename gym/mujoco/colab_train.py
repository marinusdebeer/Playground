# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os


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
""" def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give next_state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model """


""" def policy(next_state, noise_object):
    sampled_actions = tf.squeeze(actor_model(next_state))
    print(sampled_actions)
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    # return [np.squeeze(legal_action)]
    return legal_action """

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100_000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            # actor_loss = tf.math.reduce_mean(critic_value)

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


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

""" def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs *= upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model """


def policy(state):
    if np.random.random() < epsilon:
        action = np.random.uniform(low=-2.0, high=2.0, size=num_actions)
        # print("random", action)
        return action
    action = actor_model(state).numpy()[0]
    # print("purpose: ", action)
    return action




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

total_episodes = 3_000
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

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

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
critic_lr = 0.002
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
        action = policy(tf_state)
        action_end = time.time()
        action_time += action_end - action_start
        # action = policy(tf_state, ou_noise)
        next_state, reward, done, _, info = env.step(action)
        buffer.record((state, action, reward, next_state))
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