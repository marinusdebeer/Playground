import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import torch
# import tensorflow_probability as tfp
import gymnasium as gym
from collections import deque
import random
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import threading
from queue import Queue
import cv2
import math
import os
from send_email import send_email
from dotenv import load_dotenv
load_dotenv()

# https://chat.openai.com/c/2b12934e-de73-4813-a047-8db8e8aa87a7
# Hyperparameters
NUM_ACTIONS = 3
ACTIONS = [0, 2, 3]
GAMMA = 0.95
ALPHA = 0.80
BETA = 0.60
BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 80_000
N_STEPS = 3
BATCH_SIZE = 256
TRAINING_FREQ = 32
SHOW_FRAME = 100
TARGET_UPDATE_FREQ = 2500
INITIAL_LEARNING_RATE = 0.000_25 #0.00025 #0.002
LEARNING_RATE_DECAY = 0.05
EPSILON_START = 1.0
EPSILON_DECAY = 0.04

NUM_EPISODES = 5_000
SAVE_FREQ = 100
EMAIL_FREQUENCY = 1_000
# SAVE_PATH = "F:/Coding/breakout/full_rainbow/"
SAVE_PATH = "F:/Coding/breakout/3_n_steps_optimized/"
RENDER = False
PRETRAINED = False
TRAINING = True
MODEL = ""

DOUBLE_DQN                    = True
PRIORITIZED_EXPERIENCE_REPLAY = False
N_STEPS_IMPLEMENTED           = True
DUELING_DQN                   = False
DISTIBUTIONAL_RL              = False
NOISY_NETS                    = False

def decay(initial, episode, decay_rate):
    return initial * (decay_rate ** (episode/NUM_EPISODES))

def estimate_remaining_time(total_episodes, current_episode, times):
    # Fit a line to the times
    x = np.arange(current_episode)
    coefficients = np.polyfit(x, times, 1)  # Fit a line (polynomial of degree 1)
    fitted_line = np.poly1d(coefficients)

    # Predict the times of the remaining episodes
    remaining_episodes = np.arange(current_episode, total_episodes)
    predicted_times = fitted_line(remaining_episodes)

    # Sum the predicted times to get the total remaining time
    remaining_time = np.sum(predicted_times)

    return remaining_time

def read_file():
    try:
        with open("C:/Users/Marinus/OneDrive/Desktop/Development/Playground/gym/atari_breakout/rainbow_breakout.py", "r") as f:
            return f.read()
    except:
        return ""
def write_to_file(output_str, file_name="output.txt"):
    if TRAINING:
        try:
            with open(f"{SAVE_PATH}{file_name}", "a") as f:
                f.write(output_str)
        except:
            print("error writing to file")
            pass
        
def send_email_notification(all_rewards, output_str):
    try:
        def get_averages(rewards):
          averages = []
          chunk_size = 100
          for i in range(0, len(rewards), chunk_size):
              chunk = rewards[i:i + chunk_size]
              average = sum(chunk) / len(chunk)
              averages.append(average)
          return averages

        average_per_100 = get_averages(all_rewards)

        email_subject = SAVE_PATH
        email_body = f"""{output_str}\n
Average per 100: {average_per_100}\n"""
        write_to_file(email_body)
        print("\n")
        send_email(email_subject, f"{email_body}\nAll Rewards: {all_rewards}")
        print("\n")
    except Exception as e:
        print("email error", e)
        pass
    
# Prioritized experience replay buffer
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
        samples = [self.memory[i] for i in indices]

        with open(f"{SAVE_PATH}samples.txt", "a") as f:
            f.write(f"{','.join(map(str, indices))}\n")
            
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        # weights /= (len(self.memory) * probs.min()) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        # samples = [(tf.cast(state, tf.float32), action, reward, tf.cast(next_state, tf.float32), done) for state, action, reward, next_state, done in samples]
        return samples, indices, weights
        # return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
        # self.max_priority = max(self.priorities)

    def __len__(self):
        return len(self.memory)

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='relu', input_shape=(None, 84, 84, 4))
        self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = Flatten()
        if DUELING_DQN:
            self.dense1_adv = Dense(512, activation='relu')
            self.dense2_adv = Dense(NUM_ACTIONS)
            self.dense1_val = Dense(512, activation='relu')
            self.dense2_val = Dense(1)
        else:
            self.dense1 = Dense(512, activation='relu')
            self.dense2 = Dense(NUM_ACTIONS)

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        if DUELING_DQN:
            adv = self.dense1_adv(x)
            adv = self.dense2_adv(adv)

            val = self.dense1_val(x)
            val = self.dense2_val(val)

            # Combine advantage and value streams
            q_values = val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
        else:
            x = self.dense1(x)
            q_values = self.dense2(x)
        return q_values
class RainbowAgent:
    def __init__(self):
        self.frame_buffer = deque(maxlen=4)
        self.num_actions = NUM_ACTIONS
        self.learning_rate = INITIAL_LEARNING_RATE
        self.epsilon = EPSILON_START
        if PRIORITIZED_EXPERIENCE_REPLAY:
            self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)
            self.beta = BETA
            self.beta_increment = (1.0 - BETA) / NUM_EPISODES
        else:
            self.buffer = deque(maxlen=BUFFER_SIZE)
        if N_STEPS_IMPLEMENTED:
            self.n_step_buffer = deque(maxlen=N_STEPS)
            # self.n_step_buffer = ReversedDeque(maxlen=N_STEPS)
        self.fig_frame = None
        self.q_network = DQN()
        self.target_network = DQN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
        
        if RENDER:
            self.env = gym.make('Breakout-v4', render_mode="human")
        else:
            self.env = gym.make('Breakout-v4')
        if PRETRAINED:
            print(f"..................Pretrained model..................\nMODEL: {MODEL}")
            dummy_input = np.zeros((1, 84, 84, 4))
            self.q_network(dummy_input)
            self.q_network.load_weights(MODEL)
            self.target_network(dummy_input)

        human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting training at {human_readable_time}")
        write_to_file(f"Starting training at {human_readable_time}\n\n")

        model=f"""Rainbow Algorithm: https://arxiv.org/pdf/1710.02298.pdf
  - Double DQN                        - {DOUBLE_DQN}
  - Prioritized Experience Replay     - {PRIORITIZED_EXPERIENCE_REPLAY}
  - Multi-step returns                - {N_STEPS_IMPLEMENTED}
  - Dueling DQN                       - {DUELING_DQN}
  - Noisy Nets                        - {NOISY_NETS}
  - Distributional RL                 - {DISTIBUTIONAL_RL}\n\n"""
        write_to_file(model)
        print(model)
        
        params = f"""Hyperparameters:
NUM_ACTIONS = {NUM_ACTIONS}
ACTIONS = {ACTIONS}
GAMMA = {GAMMA}
ALPHA = {ALPHA}
BETA = {BETA}
BUFFER_SIZE = {BUFFER_SIZE}
N_STEPS = {N_STEPS}
BATCH_SIZE = {BATCH_SIZE}
TARGET_UPDATE_FREQ = {TARGET_UPDATE_FREQ}
TRAINING_FREQ = {TRAINING_FREQ}
SHOW_FRAME = {SHOW_FRAME}
INITIAL_LEARNING_RATE = {INITIAL_LEARNING_RATE}
LEARNING_RATE_DECAY = {LEARNING_RATE_DECAY}
EPSILON_START = {EPSILON_START}
EPSILON_DECAY = {EPSILON_DECAY}
NUM_EPISODES = {NUM_EPISODES}
SAVE_FREQ = {SAVE_FREQ}
SAVE_PATH = {SAVE_PATH}
RENDER = {RENDER}
PRETRAINED = {PRETRAINED}
TRAINING = {TRAINING}
MODEL = {MODEL}\n\n"""
        print(params)
        write_to_file(f"\n\n\n{read_file()}\n\n\n")
        self.update_target_network()

    def update_target_network(self):
        if TRAINING:
            print("updating target network")
            write_to_file("updating target network\n")
            self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if N_STEPS_IMPLEMENTED:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) < N_STEPS:
                return
            n_step_reward, n_step_state, n_step_done = self.calculate_n_step_info()
            state, action, _, _, _ = self.n_step_buffer[0]
            if PRIORITIZED_EXPERIENCE_REPLAY:
                self.buffer.add(state, action, n_step_reward, n_step_state, n_step_done)
            else:
                self.buffer.append((state, action, n_step_reward, n_step_state, n_step_done))
        else:
            if PRIORITIZED_EXPERIENCE_REPLAY:
                self.buffer.add(state, action, reward, next_state, done)
            else:
                self.buffer.append((state, action, reward, next_state, done))

    def calculate_n_step_info(self):
        n_step_reward = 0
        n_step_state = self.n_step_buffer[-1][-2]  # default to the last state in the buffer
        n_step_done = self.n_step_buffer[-1][-1]  # default to the last done flag in the buffer

        for idx, (_, _, reward, next, done) in enumerate(reversed(self.n_step_buffer)):
            n_step_reward += (GAMMA ** idx) * reward
            if done:
                n_step_done = True # if a terminal state is encountered, update the n_step_done flag
                break
        return n_step_reward, n_step_state, n_step_done
    
    # Use tf.function to speed up the computation graph
    @tf.function
    def compute_loss_and_gradients(self, states, actions, rewards, next_states, dones, weights=None):
        # print("computing loss and gradients")
        # print(type(states))
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)

            next_q_values = self.target_network(next_states)
            next_q_values_online = self.q_network(next_states)

            next_actions = tf.argmax(next_q_values_online, axis=1)
            next_q_values = tf.reduce_sum(next_q_values * tf.one_hot(next_actions, self.num_actions), axis=1)

            target_q_values = rewards + GAMMA * (1 - dones) * next_q_values
            target_q_values = tf.stop_gradient(target_q_values)
            td_errors = target_q_values - q_values
            if PRIORITIZED_EXPERIENCE_REPLAY:
                # loss = tf.reduce_mean(weights * tf.square(td_errors) * 0.5)
                loss = tf.keras.losses.MSE(target_q_values, q_values)
                loss = tf.reduce_mean(weights * loss)
            else:
                loss = tf.keras.losses.MSE(target_q_values, q_values)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        return loss, gradients, td_errors

    def learn(self):
        if len(self.buffer) < MIN_BUFFER_SIZE:
            return 0, 0
        start = time.time()
        if PRIORITIZED_EXPERIENCE_REPLAY:
            minibatch, indices, weights = self.buffer.sample(BATCH_SIZE, self.beta)
        else:
            minibatch = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        # states, actions, rewards, next_states, dones = map(tf.stack, zip(*minibatch))

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        if PRIORITIZED_EXPERIENCE_REPLAY:
            loss, gradients, td_errors = self.compute_loss_and_gradients(states, actions, rewards, next_states, dones, weights)
            priorities = tf.abs(td_errors) + 1e-6
            self.buffer.update_priorities(indices, priorities.numpy())
        else:
            loss, gradients, td_errors = self.compute_loss_and_gradients(states, actions, rewards, next_states, dones, None)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        end = time.time()
        return round(end - start, 2), loss.numpy()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon and TRAINING:
            return np.random.choice(ACTIONS)
        else:
            q_values = self.q_network(np.array([state], dtype=np.float32))
            action = ACTIONS[np.argmax(q_values.numpy())]
        return action
    
    def show_frame(self, state):
        if self.fig_frame is None or not plt.fignum_exists(self.fig_frame.number):
            self.fig_frame = plt.figure("rainbow.py") 
            self.ax_frame = self.fig_frame.add_subplot(111)
        self.ax_frame.clear()
        self.ax_frame.imshow(state, cmap='gray')
        self.fig_frame.canvas.draw()
        plt.pause(0.01)
    
    def preprocess_state(self, state):
        start = time.time()
        try:
            state.shape
        except:
            state = state[0]
            pass
        processed_observe = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        processed_observe = cv2.resize(processed_observe, (84, 84), interpolation=cv2.INTER_AREA)
        return processed_observe, time.time() - start
    
    def get_frames(self):
        state = self.env.reset()
        steps = 0
        lives = 5
        for _ in range(4):
            processed_state, proc_time = self.preprocess_state(state)
            self.frame_buffer.append(processed_state)
        state = np.stack(self.frame_buffer, axis=-1)

        pbar = tqdm(total=MIN_BUFFER_SIZE)
        while steps < MIN_BUFFER_SIZE:
            action = np.random.choice(ACTIONS)
            next_state, reward, done, _, info = self.env.step(action)

            if action != 0:
                action -= 1
            if (info["lives"] < lives):
                lives = info["lives"]
                reward = -15
                self.env.step(1)
            processed_state, proc_time = self.preprocess_state(next_state)
            self.frame_buffer.append(processed_state)
            next_state = np.stack(self.frame_buffer, axis=-1)

            self.remember(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            pbar.update(1)
        pbar.close()
        write_to_file(f"Finished getting {steps} frames\n")
        print(f"Finished getting {steps} frames")

    def train(self):

        steps = 0
        # num_threads = 4
        # sync_queue = Queue()
        # lock = threading.Lock()
        # threads = []
        all_rewards = []
        ep_times = []
        training_loss = []
        total_actions = {0: 0, 2: 0, 3: 0}
        highscore = 0
        self.get_frames()
        start_time = time.time()
        human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting training at {human_readable_time}")
        write_to_file(f"Starting training at {human_readable_time}\n")
        # for _ in range(num_threads):
        #     thread = threading.Thread(target=self.learn(), args=(agent, sync_queue, lock))
        #     thread.start()
        #     threads.append(thread)
        for episode in range(1, NUM_EPISODES + 1):
            ep_start = time.time()
            actions = {0: 0, 2: 0, 3: 0}
            done = False
            ep_reward = 0
            lives = 5
            state = self.env.reset()
            for _ in range(4):
                processed_state, proc_time = self.preprocess_state(state)
                self.frame_buffer.append(processed_state)
            state = np.stack(self.frame_buffer, axis=-1)
            training_time = 0
            total_actions_time = 0
            step_times = 0
            procs_time = 0
            self.env.step(1)
            while not done:
                action_time = time.time()
                action = self.choose_action(state)
                total_actions_time += time.time() - action_time
                actions[action] += 1
                total_actions[action] += 1
                step_time = time.time()
                next_state, reward, done, _, info = self.env.step(action)
                step_times += time.time() - step_time

                if action != 0:
                    action -= 1
                processed_state, proc_time = self.preprocess_state(next_state)
                procs_time += proc_time
                self.frame_buffer.append(processed_state)
                next_state = np.stack(self.frame_buffer, axis=-1)
                steps += 1
                if steps % SHOW_FRAME == 0:
                    self.show_frame(processed_state)
                if steps % TRAINING_FREQ == 0 and TRAINING:
                    learn_time, loss = self.learn()
                    training_time += learn_time
                    if steps > TARGET_UPDATE_FREQ:
                        training_loss.append(loss)
                ep_reward += reward

                if (info["lives"] < lives):
                    lives = info["lives"]
                    reward = -15
                    self.env.step(1)
                    self.remember(state, action, reward, next_state, True)
                elif TRAINING:
                    self.remember(state, action, reward, next_state, done)
                state = next_state

                if steps % TARGET_UPDATE_FREQ == 0:
                    self.update_target_network()

                if done:
                    all_rewards.append(ep_reward)
                    if ep_reward > highscore:
                        highscore = ep_reward
                    elapsed_time = round(time.time() - start_time, 1)
                    ep_times.append(time.time() - ep_start)

                    avgTime = round(elapsed_time / episode, 1)
                    ep_time = round(time.time() - ep_start, 1)
                    avg100time = np.round(np.mean(ep_times[-100:]), 1)
                    avg10 = np.round(np.mean(all_rewards[-10:]), 1)
                    avg100 = np.round(np.mean(all_rewards[-100:]), 1)
                    avg500 = np.round(np.mean(all_rewards[-500:]), 1)
                    avg1000 = np.round(np.mean(all_rewards[-1000:]), 1)
                    avg = np.round(np.mean(all_rewards), 1)

                    lossAvg = np.round(np.mean(training_loss), 5)
                    loss1K = np.round(np.mean(training_loss[-1_000:]), 5)
                    loss10K = np.round(np.mean(training_loss[-10_000:]), 5)
                    loss100K = np.round(np.mean(training_loss[-100_000:]), 5)
                    loss1M = np.round(np.mean(training_loss[-1_000_000:]), 5)

                    actions_per_episode = actions[0] + actions[2] + actions[3]
                    actions_per_training = total_actions[0] + total_actions[2] + total_actions[3]
                    if total_actions_time != 0:
                        actions_per_second = round(actions_per_episode / total_actions_time)
                    else:
                        actions_per_second = 'lots'

                    output_str = f"Episode {episode}/{NUM_EPISODES}, Highscore: {highscore}, Reward: {ep_reward}, Epsilon: {round(self.epsilon, 3)}, LR: {round(self.learning_rate, 6)}, NAME: {SAVE_PATH}\n"
                    output_str += f"Average: {avg}, Avg10: {avg10}, Avg100: {avg100}, Avg500: {avg500}, avg1000: {avg1000}, beta: {getattr(self, 'beta', 'N/A')}\n"
                    output_str += f"lossAvg: {lossAvg}, loss1K: {loss1K}, loss10K: {loss10K}, loss100K: {loss100K}, loss1M: {loss1M}\n"
                    output_str += f"Total time: {elapsed_time}s, Episode time: {ep_time}s, Average time: {avgTime}s, Avg100: {avg100time}\n"
                    output_str += f"No op: {actions[0]}/{total_actions[0]}, Left: {actions[3]}/{total_actions[3]}, Right: {actions[2]}/{total_actions[2]}, Total: {actions_per_episode}/{actions_per_training}, memory size: {len(self.buffer)}\n"
                    output_str += f"Training time: {round(training_time, 1)}s, "
                    output_str += f"Action time: {round(total_actions_time, 1)}s, Actions per second: {actions_per_second}, "
                    output_str += f"Step time: {round(step_times, 1)}s, "
                    output_str += f"Preprocessing time: {round(procs_time, 1)}s, "
                    output_str += f"Total time: {round(training_time + total_actions_time + step_times + procs_time, 1)}/{round(time.time() - ep_start, 1)}s\n"

                    if episode % SAVE_FREQ == 0 and TRAINING:
                        output_str += f"Model weights saved at episode {episode}\n"
                        self.q_network.save_weights(f"{SAVE_PATH}models/model_{episode}.h5")

                        time_remaining = estimate_remaining_time(NUM_EPISODES, episode, ep_times)
                        finish_time = datetime.now() + timedelta(seconds=time_remaining)
                        finish_time = finish_time.strftime("%Y-%m-%d %H:%M:%S")
                        output_str += f"\nEstimated finish time: {finish_time}, which is in{round(time_remaining)}s\n"

                        with open(f"{SAVE_PATH}rewards.txt", "w") as f:
                            f.write(str(all_rewards))
                        with open(f"{SAVE_PATH}loss.txt", "w") as f:
                            f.write(str(training_loss))
                    if episode % EMAIL_FREQUENCY == 0 or episode == 1:
                        send_email_notification(all_rewards, output_str)
                    write_to_file(f"{output_str}\n")
                    print(output_str, end='\n')
            # if steps == MIN_BUFFER_SIZE:
            # self.epsilon = EPSILON_START - (episode/(NUM_EPISODES*(1/EPSILON_START)))
            # self.epsilon = EPSILON_START * (EPSILON_DECAY ** episode)
            if getattr(self, 'beta', False):
                self.beta = min(1.0, self.beta + self.beta_increment)
            self.epsilon = decay(EPSILON_START, episode, EPSILON_DECAY)
            # self.learning_rate = decay(INITIAL_LEARNING_RATE, episode, LEARNING_RATE_DECAY)
            # self.optimizer.learning_rate = self.learning_rate

if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        os.makedirs(f"{SAVE_PATH}models")
    agent = RainbowAgent()
    agent.train()
    human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished training at {human_readable_time}")
    write_to_file(f"Finished training at {human_readable_time}\n")