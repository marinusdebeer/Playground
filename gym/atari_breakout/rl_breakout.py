import numpy as np
import gymnasium as gym
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from collections import deque
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
import threading


class DQN(tf.keras.Model):
    def __init__(self, num_actions, model_truncate=0):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(
            32, 8, strides=4, activation='relu', input_shape=(84-model_truncate, 84, 4))
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
    def __init__(self, num_actions=3):
        self.save_path = "models_3_actions_small_reward_noop/"
        self.model_truncate = 50

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

        self.model = DQN(self.num_actions, self.model_truncate)
        self.target_model = DQN(self.num_actions, self.model_truncate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # MODEL = f"models_3_actions/model_weights_episode_5000.h5"
        # dummy_input = np.zeros((1, 84-self.model_truncate, 84, 4))
        # self.model(dummy_input)
        # self.model.load_weights(MODEL)
        # self.target_model(dummy_input)

        self.update_target_network()

    def preprocess_state(self, state):
        try:
            state.shape
        except:
            state = state[0]
            pass
        gray = rgb2gray(state)
        resized = resize(gray, (84, 84), mode='constant')
        resized = resized[self.model_truncate:]
        processed_observe = np.uint8(resized * 255)
        # rows, cols = np.nonzero(processed_observe)
        # for row, col in zip(rows, cols):
        # val = processed_observe[row, col]
        # print(f"Value {val} found at index ({row}, {col})")
        return processed_observe

    def show_frame(self, state):
        plt.imshow(state, cmap='gray')
        fig = plt.show(block=False)
        plt.pause(0.01)  # Pause for a brief period
        plt.clf()
        # plt.close(fig)

    def display_stacked_frames(self, state, pause_time=0.5):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i in range(4):
            frame = state[:, :, i]
            axes[i].imshow(frame, cmap='gray')
            axes[i].axis('off')

        plt.show(block=False)
        plt.pause(pause_time)
        plt.close(fig)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.model(np.array([state], dtype=np.float32))
        action = np.argmax(q_values.numpy()[0])
        if action != 0:
            action += 1
        return action

    def update_target_network(self):
        print("updating target network")
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, lock):
        if len(self.memory) < self.batch_size:
            return
        # print("training")
        with lock:
            minibatch = random.sample(self.memory, self.batch_size)

            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            with tf.GradientTape() as tape:
                q_values = self.model(states)
                q_values_next = self.target_model(next_states)
                target_q_values = rewards + self.gamma * \
                    np.amax(q_values_next, axis=1) * (1 - dones)
                target_q_values = tf.stop_gradient(target_q_values)
                q_values_pred = tf.reduce_sum(
                    q_values * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.keras.losses.MSE(target_q_values, q_values_pred)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

    def train(self):
        if self.render:
            env = gym.make('Breakout-v4', render_mode="human")
        else:
            env = gym.make('Breakout-v4')
            # env = gym.make('BreakoutDeterministic-v4')
        step_count = 0
        highscore = 0
        total_action_counter = {0: 0, 2: 0, 3: 0}
        replay_lock = threading.Lock()
        for episode in tqdm(range(1, self.num_episodes + 1), ascii=True, unit='episodes'):
            lives = 5
            blocks = 0
            action_counter = {0: 0, 2: 0, 3: 0}
            state = env.reset()
            # learning_rate = self.initial_learning_rate * self.decay_rate ** (episode / self.num_episodes)
            # self.optimizer.learning_rate = learning_rate
            for _ in range(4):
                self.frame_buffer.append(self.preprocess_state(state))
            state = np.stack(self.frame_buffer, axis=-1)
            done = False
            episode_reward = 0
            env.step(1)
            while not done:
                action = self.choose_action(state)
                action_counter[action] += 1
                total_action_counter[action] += 1
                next_state, reward, done, _, info = env.step(action)
                blocks += reward
                reward *= 20
                if action != 0:
                    action -= 1
                else:
                    reward += 1
                # self.show_frame(self.preprocess_state(next_state))
                if step_count % self.training_frequency == 0:
                    self.show_frame(self.preprocess_state(next_state))
                    replay_thread = threading.Thread(
                        target=self.replay, args=(replay_lock,))
                    replay_thread.start()
                    # self.replay()
                self.frame_buffer.append(self.preprocess_state(next_state))
                next_state = np.stack(self.frame_buffer, axis=-1)
                # self.display_stacked_frames(next_state)

                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step_count += 1

                if (info["lives"] < lives):
                    lives = info["lives"]
                    env.step(1)
                if done:
                    if blocks > highscore:
                        highscore = blocks
                    print(
                        f"Episode {episode}, Blocks: {blocks}, Highscore: {highscore}, Reward: {episode_reward}, Epsilon: {self.epsilon}, No op: {action_counter[0]}, Left: {action_counter[3]}, Right: {action_counter[2]}")
                    print(
                        f"Total No op: {total_action_counter[0]}, Total Left: {total_action_counter[3]}, Total Right: {total_action_counter[2]}")
                if step_count % self.target_update_freq_steps == 0:
                    self.update_target_network()

            self.epsilon = pow(self.epsilon_decay, episode)

            if episode % 10 == 0 or episode == 0:
                self.model.save_weights(
                    f"{self.save_path}model_weights_episode_{episode}.h5")
                print(f"Model weights saved at episode {episode}")


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()