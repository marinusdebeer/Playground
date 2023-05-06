import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.models import Model
import gym
from collections import deque
import random

# Hyperparameters
NUM_ACTIONS = 4
GAMMA = 0.99
ALPHA = 0.6
BETA = 0.4
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1_000
LEARNING_RATE = 0.000_1
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10_000
NUM_EPISODES = 1_000

# Prioritized experience replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# Create the Q-Network model with dueling architecture
def create_q_network(num_actions):
    input_state = Input(shape=(4, 84, 84))
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_state)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    value_fc = Dense(512, activation='relu')(x)
    value = Dense(1, activation=None)(value_fc)

    advantage_fc = Dense(512, activation='relu')(x)
    advantage = Dense(num_actions, activation=None)(advantage_fc)

    q_values = Add()([value, advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True)])

    model = Model(inputs=input_state, outputs=q_values)
    return model

# Rainbow agent
class RainbowAgent:
    def __init__(self, num_actions, learning_rate):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = EPSILON_START
        self.epsilon_decay =        (EPSILON_START - EPSILON_END) / EPSILON_DECAY
        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)
        self.beta = BETA
        self.beta_increment = (1.0 - BETA) / NUM_EPISODES

        self.q_network = create_q_network(num_actions)
        self.target_network = create_q_network(num_actions)
        self.update_target_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_network(np.expand_dims(state, axis=0))
            return np.argmax(q_values.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        self.beta = min(1.0, self.beta + self.beta_increment)

        samples, indices, weights = self.buffer.sample(BATCH_SIZE, self.beta)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)

            next_q_values = self.target_network(next_states)
            next_q_values_online = self.q_network(next_states)

            next_actions = tf.argmax(next_q_values_online, axis=1)
            next_q_values = tf.reduce_sum(next_q_values * tf.one_hot(next_actions, self.num_actions), axis=1)

            target_q_values = rewards + GAMMA * (1 - dones) * next_q_values
            td_errors = target_q_values - q_values
            loss = tf.reduce_mean(weights * tf.square(td_errors) * 0.5)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        priorities = np.abs(td_errors.numpy()) + 1e-6
        self.buffer.update_priorities(indices, priorities)

    def train(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        for episode in range(NUM_EPISODES):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                self.remember(state, action, reward, next_state, done)
                self.learn()
                state = next_state

                self.epsilon -= self.epsilon_decay

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

            if (episode + 1) % TARGET_UPDATE_FREQ == 0:
                self.update_target_network()

if __name__ == "__main__":
    agent = RainbowAgent(NUM_ACTIONS, LEARNING_RATE)
    agent.train()

