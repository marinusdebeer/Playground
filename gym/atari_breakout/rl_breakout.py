import numpy as np
import gymnasium as gym
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from collections import deque
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
import threading
import pickle
import time
from send_email import send_email

BUFFER_SIZE = 250_000
ALPHA = 0.6
BETA = 0.4
BATCH_SIZE = 1024
NUM_ACTIONS = 3
# self.epsilon_decay = 0.9997
EPSILON_START = 0.10
GAMMA = 0.99

LEARNING_RATE = 0.000_25
NUM_EPISODES = 500
TARGET_UPDATE_FREQ= 5_000
TRAINING_FREQ = 200
RENDER = True
ACTIONS = [0, 2, 3]
TRAINING = False
PRETRAINED = True
# MODEL="models_3_actions/best_models_ranked_1_to_best/model_430.h5"
MODEL="models_3_actions/models/model_430.h5"
SAVE_PATH = "models_3_actions/"
EXPERIENCE_REPLAY = "models_3_actions/experience_replay.pkl"
SAVE_FREQ=10
def write_to_file(output_str, file_name="output.txt"):
    if TRAINING:
        with open(f"{SAVE_PATH}{file_name}", "a") as f:
            f.write(output_str)

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='relu', input_shape=(84, 84, 4))
        self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(NUM_ACTIONS)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha):
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        indices.sort()
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= (len(self.memory) * probs.min()) ** (-beta)
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)
        self.frame_buffer = deque(maxlen=4)
        self.epsilon = EPSILON_START
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.beta = BETA
        self.beta_increment = (1.0 - BETA) / NUM_EPISODES
        self.fig_frame = None
        if PRETRAINED:
            print("..................Pretrained model..................")
            dummy_input = np.zeros((1, 84, 84, 4))
            self.model(dummy_input)
            self.model.load_weights(MODEL)
            self.target_model(dummy_input)
            self.update_target_network()

        if RENDER:
            self.env = gym.make('Breakout-v4', render_mode="human")
        else:
            self.env = gym.make('Breakout-v4')

        params = f"""Hyperparameters:
NUM_ACTIONS = {NUM_ACTIONS}
ACTIONS = {ACTIONS}
GAMMA = {GAMMA}
ALPHA = {ALPHA}
BETA = {BETA}
BUFFER_SIZE = {BUFFER_SIZE}
BATCH_SIZE = {BATCH_SIZE}
TARGET_UPDATE_FREQ = {TARGET_UPDATE_FREQ}
TRAINING_FREQ = {TRAINING_FREQ}
LEARNING_RATE = {LEARNING_RATE}
EPSILON_START = {EPSILON_START}
NUM_EPISODES = {NUM_EPISODES}
SAVE_FREQ = {SAVE_FREQ}
SAVE_PATH = {SAVE_PATH}
RENDER = {RENDER}
PRETRAINED = {PRETRAINED}
MODEL = {MODEL}\n\n"""
        write_to_file(params, "output.txt")
        print(params)
        # self.fig_scatter, self.scatter_axes = plt.subplots()
        # self.scatter_axes.set_xlabel('Episodes')
        # self.scatter_axes.set_ylabel('Average Rewards Per 10 Episodes')
        # self.scatter_axes.set_title('Episode vs Average Rewards Per 10 Episodes')
        self.rewards = []
        self.episodes = []
    def preprocess_state(self, state):
        try:
            state.shape
        except:
            state = state[0]
            pass
        gray = rgb2gray(state)
        resized = resize(gray, (84, 84), mode='constant')
        processed_observe = np.uint8(resized * 255)
        return processed_observe

    def show_frame(self, state):
        if self.fig_frame is None or not plt.fignum_exists(self.fig_frame.number):
            self.fig_frame = plt.figure("rl_breakout.py") 
            self.ax_frame = self.fig_frame.add_subplot(111)
        self.ax_frame.clear()
        self.ax_frame.imshow(state, cmap='gray')
        self.fig_frame.canvas.draw()
        plt.pause(0.01)
        
    def update_scatter_plot(self, episode, reward):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.scatter_axes.plot(self.episodes, self.rewards, 'b')
        # self.scatter_axes.scatter(self.episodes, self.rewards, marker='o', color='b')
        self.scatter_axes.relim()
        self.scatter_axes.autoscale_view()
        self.fig_scatter.canvas.draw()
        plt.pause(0.01)

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
        self.memory.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon and TRAINING:
            return np.random.choice(ACTIONS)
        
        q_values = self.model(np.array([state], dtype=np.float32))
        action = np.argmax(q_values.numpy()[0])
        if action != 0:
            action += 1
        return action

    def update_target_network(self):
        print("updating target network")
        write_to_file("updating target network\n")
        self.target_model.set_weights(self.model.get_weights())
    # Load experience_replay from file
    def load_experience_replay(self):
        with open(EXPERIENCE_REPLAY, 'rb') as f:
            self.memory = pickle.load(f)

    def save_experience_replay(self):
        print("saving experience replay")
        with open(EXPERIENCE_REPLAY, 'wb') as f:
            pickle.dump(self.memory, f)

    def replay(self, lock=None):
        if len(self.memory) < BATCH_SIZE:
            return
        # minibatch = random.sample(self.memory, self.batch_size)
        self.beta = min(1.0, self.beta + self.beta_increment)
        minibatch, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, NUM_ACTIONS), axis=1)

            next_q_values = self.target_model(next_states)
            next_q_values_online = self.model(next_states)

            next_actions = tf.argmax(next_q_values_online, axis=1)
            next_q_values = tf.reduce_sum(next_q_values * tf.one_hot(next_actions, NUM_ACTIONS), axis=1)

            target_q_values = rewards + GAMMA * (1 - dones) * next_q_values
            target_q_values = tf.stop_gradient(target_q_values)
            td_errors = target_q_values - q_values
            loss = tf.reduce_mean(weights * tf.square(td_errors) * 0.5)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        priorities = tf.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, priorities.numpy())

    def train(self):
        step_count = 0
        highscore = 0
        total_action_counter = {0: 0, 2: 0, 3: 0}
        blocks = 0
        all_rewards = []
        start_time = time.time()
        write_to_file(f"Starting training at {start_time}\n")
        # replay_lock = threading.Lock()
        for episode in range(1, NUM_EPISODES + 1):
            ep_start = time.time()
            lives = 5
            action_counter = {0: 0, 2: 0, 3: 0}
            done = False
            episode_reward = 0
            state = self.env.reset()
            # learning_rate = self.initial_learning_rate * self.decay_rate ** (episode / NUM_EPISODES)
            # self.optimizer.learning_rate = learning_rate
            for _ in range(4):
                self.frame_buffer.append(self.preprocess_state(state))
            state = np.stack(self.frame_buffer, axis=-1)
            
            self.env.step(1)
            while not done:
                action = self.choose_action(state)
                action_counter[action] += 1
                total_action_counter[action] += 1
                next_state, reward, done, _, info = self.env.step(action)
                if action != 0:
                    action -= 1
                if step_count % TRAINING_FREQ == 0:
                    self.show_frame(self.preprocess_state(next_state))
                    if TRAINING:
                        self.replay()
                    # replay_thread = threading.Thread(target=self.replay, args=(replay_lock,))
                    # replay_thread.start()
                self.frame_buffer.append(self.preprocess_state(next_state))
                next_state = np.stack(self.frame_buffer, axis=-1)
                episode_reward += reward
                
                step_count += 1

                if (info["lives"] < lives):
                    lives = info["lives"]
                    reward = -15
                    # print(f"lost a life and got {reward} points, done: {done}")
                    self.env.step(1)
                if TRAINING:
                    self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    all_rewards.append(episode_reward)
                    blocks += episode_reward
                    if episode_reward > highscore:
                        highscore = episode_reward
                    elapsed_time = time.time() - start_time
                    average_time_per_episode = elapsed_time / episode
                    output_str = f"Episode {episode}/{NUM_EPISODES}, Highscore: {highscore}, Reward: {episode_reward}, Epsilon: {round(self.epsilon, 3)}\n"
                    output_str += f"Total time: {round(elapsed_time, 1)}s, Episode time: {round(time.time() - ep_start, 1)}s, Average time: {round(average_time_per_episode, 1)}s\n"
                    output_str += f"No op: {action_counter[0]}, Left: {action_counter[3]}, Right: {action_counter[2]}, Total: {action_counter[0] + action_counter[2] + action_counter[3]}\n"
                    output_str += f"Total No op: {total_action_counter[0]}, Total Left: {total_action_counter[3]}, Total Right: {total_action_counter[2]}, Total: {total_action_counter[0] + total_action_counter[2] + total_action_counter[3]}, memory size: {len(self.memory)}\n\n"
                    write_to_file(output_str)
                    print(output_str, end='')
                    # print(f"Episode {episode}, Highscore: {highscore}, Reward: {episode_reward}, Epsilon: {self.epsilon}")
                    # print(f"Total No op: {total_action_counter[0]}, Total Left: {total_action_counter[3]}, Total Right: {total_action_counter[2]}, memory size: {len(self.memory)}")
                if step_count % TARGET_UPDATE_FREQ == 0 and TRAINING:
                    self.update_target_network()

            self.epsilon = EPSILON_START -(episode/(NUM_EPISODES*(1/EPSILON_START)))
            if episode % SAVE_FREQ == 0 and TRAINING:
                self.model.save_weights(f"{SAVE_PATH}models/model_{episode}.h5")
                # send_email(f"Average Rewards for ep: {episode}: {round(blocks/10)}", f"Total: {blocks}, Highscore: {highscore}")
                print(f"Model weights saved at episode {episode}")
                # self.update_scatter_plot(episode, blocks/10)
                # self.fig_scatter.savefig(f'models_3_actions/episode{episode}_vs_rewards.png', dpi=300)
                blocks = 0
                with open(f"{SAVE_PATH}rewards.txt", "w") as f:
                    f.write(str(all_rewards))
            # if episode % 500 == 0 and TRAINING:  
                # start = time.time()
                # self.save_experience_replay()
                # print(f"Experience replay saved at episode {episode}, took {time.time() - start} seconds")


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()