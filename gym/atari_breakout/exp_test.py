import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tqdm import tqdm




class DQN(tf.keras.Model):
    def __init__(self, num_actions, model_truncate=0):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='relu', input_shape=(84-model_truncate, 84, 4))
        self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class DQNAgent:
    def __init__(self, num_actions=3, model_truncate=0, trained=False):
        self.num_actions = num_actions
        self.memory = deque(maxlen=100_000)
        self.frame_buffer = deque(maxlen=4)
        self.epsilon = 1
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.00025
        # self.initial_learning_rate = 0.001
        self.decay_rate = 0.99

        self.num_episodes = 5000
        self.target_update_freq_steps = 3_000
        self.batch_size = 512
        self.training_frequency = 100
        self.render = False
        self.actions = [0, 2, 3]

        self.model_truncate = model_truncate
        self.model = DQN(self.num_actions, self.model_truncate)
        self.target_model = DQN(self.num_actions, self.model_truncate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.save_path = ""
        # dummy_input = np.zeros((1, 84-self.model_truncate, 84, 4))
        # self.model(dummy_input)
        # self.model.load_weights(f"model_weights_episode_350.h5")
        # self.target_model(dummy_input)

        self.update_target_network()
    def load_experiences(self):
        with open("experience_replay.pkl", 'rb') as f:
            self.memory = pickle.load(f)
            print(f"Loaded {len(self.memory)} experiences")
    def show_stacked_frames(self, states):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for state in states:
            for i in range(4):
                frame = state[:, :, i]
                axes[i].imshow(frame, cmap='gray')
                axes[i].axis('off')

            plt.show(block=False)
            plt.pause(0.2)

    def show_state(self, states):
        for state in states:
            plt.imshow(state, cmap='gray')
            fig = plt.show(block=False)
            plt.pause(0.01)  # Pause for a brief period
            plt.clf()

    def update_target_network(self):
        print("updating target network")
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, lock=None):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array([state[self.model_truncate:, :, :] for state in states], dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array([state[self.model_truncate:, :, :] for state in next_states], dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values_next = self.target_model(next_states)
            target_q_values = rewards + self.gamma * np.amax(q_values_next, axis=1) * (1 - dones)
            target_q_values = tf.stop_gradient(target_q_values)
            q_values_pred = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    def train(self):
        # training loop using tqdm
        for episode in tqdm(range(1, self.num_episodes + 1), ascii=True, unit='episodes'):
            # print(f"Episode {episode}/{self.num_episodes}")
            self.replay()
            if episode % 10 == 0:
                self.update_target_network()
                self.model.save_weights(f"{self.save_path}trained_{episode}.h5")
        self.model.summary()

if __name__ == "__main__":
    agent = DQNAgent()
    agent.load_experiences()
    agent.train()
    
    

  