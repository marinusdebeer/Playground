import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.models import Model
import gymnasium as gym
from collections import deque
import random
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import threading

# https://chat.openai.com/c/2b12934e-de73-4813-a047-8db8e8aa87a7
# Hyperparameters
NUM_ACTIONS = 3
ACTIONS = [0, 2, 3]
GAMMA = 0.99
ALPHA = 0.60
BETA = 0.40
BUFFER_SIZE = 250_000
N_STEPS = 5
BATCH_SIZE = 1024
TARGET_UPDATE_FREQ = 3_000
TRAINING_FREQ = 200
LEARNING_RATE = 0.000_25
EPSILON_START = 0.5

NUM_EPISODES = 100
SAVE_FREQ = 10
SAVE_PATH = "rainbow_models/"
RENDER = False
PRETRAINED = True
MODEL = "rainbow_models/good_models/model_10000_1.h5"
training = False
def write_to_file(output_str, file_name="output.txt"):
    global training
    if training:
        with open(f"{SAVE_PATH}{file_name}", "a") as f:
            f.write(output_str)
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
        # top_10_indices = np.argsort(probs)[::-1][:10]
        # top_10 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:10]
        # print(top_10)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)

# Create the Q-Network model with dueling architecture
def DQN(num_actions):
    input_state = Input(shape=(84, 84, 4))
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
    def __init__(self, TRAIN=False):
        global training
        training = TRAIN
        print(training)
        self.frame_buffer = deque(maxlen=4)
        self.num_actions = NUM_ACTIONS
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON_START
        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)
        self.n_step_buffer = deque(maxlen=N_STEPS)
        self.beta = BETA
        self.beta_increment = (1.0 - BETA) / NUM_EPISODES

        self.q_network = DQN(NUM_ACTIONS)
        self.target_network = DQN(NUM_ACTIONS)
        self.update_target_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        self.fig_frame, self.ax_frame = plt.subplots()
        if RENDER:
            self.env = gym.make('Breakout-v4', render_mode="human")
        else:
            self.env = gym.make('Breakout-v4')
        if PRETRAINED and training:
            print(f"..................Pretrained model..................\n{MODEL}")
            write_to_file(f"..................Pretrained model..................\n{MODEL}\n")
            dummy_input = np.zeros((1, 84, 84, 4))
            self.q_network(dummy_input)
            self.q_network.load_weights(MODEL)
            self.target_network(dummy_input)
            self.update_target_network()
        # log the hyperparameters
        write_to_file(f"""Hyperparameters:
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
        LEARNING_RATE = {LEARNING_RATE}
        EPSILON_START = {EPSILON_START}
        NUM_EPISODES = {NUM_EPISODES}
        SAVE_FREQ = {SAVE_FREQ}
        SAVE_PATH = {SAVE_PATH}
        RENDER = {RENDER}
        PRETRAINED = {PRETRAINED}
        MODEL = {MODEL}\n\n""")

    def update_target_network(self):
        print("updating target network")
        write_to_file("updating target network\n")
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        # self.n_step_buffer.append((state, action, reward, next_state, done))
        # if len(self.n_step_buffer) < N_STEPS:
        #     return
        # n_step_reward, n_step_state, n_step_done = self.calculate_n_step_info()
        # state, action, _, _, _ = self.n_step_buffer[0]
        # self.buffer.add(state, action, n_step_reward, n_step_state, n_step_done)

    def calculate_n_step_info(self):
        n_step_reward = 0
        n_step_state = self.n_step_buffer[-1][-2]
        n_step_done = self.n_step_buffer[-1][-1]

        for idx, transition in enumerate(reversed(self.n_step_buffer)):
            _, _, reward, _, done = transition
            n_step_reward += (GAMMA ** idx) * reward

            if not done:
                break

        return n_step_reward, n_step_state, n_step_done
    def learn(self, lock=None, steps=None):
        if len(self.buffer) < BATCH_SIZE:
            return
      
        # lock.acquire()
        # print("learning")
        self.beta = min(1.0, self.beta + self.beta_increment)

        samples, indices, weights = self.buffer.sample(BATCH_SIZE, self.beta)
        if steps % TRAINING_FREQ*100 == 0:
            write_to_file({samples, indices, weights}, "sample.txt")
            write_to_file("\n", "sample.txt")
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
        # lock.release()
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)
        
        q_values = self.q_network(np.expand_dims(state, axis=0))
        action = np.argmax(q_values.numpy())
        if action != 0:
            action += 1
        return action
    
    def show_frame(self, state):
        self.ax_frame.clear()
        self.ax_frame.imshow(state, cmap='gray')
        self.fig_frame.canvas.draw()
        plt.pause(0.01)

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
    
    def train(self):
        steps = 0
        all_rewards = []
        total_action_counter = {0: 0, 2: 0, 3: 0}
        highscore = 0
        start_time = time.time()
        # lock = threading.Lock()  # create a lock object
        for episode in range(1, NUM_EPISODES + 1):
            ep_start = time.time()
            action_counter = {0: 0, 2: 0, 3: 0}
            done = False
            ep_reward = 0
            lives = 5
            state = self.env.reset()
            for _ in range(4):
                self.frame_buffer.append(self.preprocess_state(state))
            state = np.stack(self.frame_buffer, axis=-1)
            
            self.env.step(1)
            while not done:
                action = self.choose_action(state)
                # print(action)
                action_counter[action] += 1
                total_action_counter[action] += 1
                next_state, reward, done, _, info = self.env.step(action)
                if action != 0:
                    action -= 1
                processed_state = self.preprocess_state(next_state)
                self.frame_buffer.append(processed_state)
                next_state = np.stack(self.frame_buffer, axis=-1)
                steps += 1
                if steps % TRAINING_FREQ == 0:
                    self.show_frame(processed_state)
                    self.learn(steps=steps)
                    # learn_thread = threading.Thread(target=self.learn, args=(lock,))
                    # learn_thread.start()
                ep_reward += reward

                if (info["lives"] < lives):
                    lives = info["lives"]
                    reward = -15
                    self.env.step(1)

                self.remember(state, action, reward, next_state, done)
                state = next_state

                if steps % TARGET_UPDATE_FREQ == 0:
                    self.update_target_network()

                if done:
                    all_rewards.append(ep_reward)
                    if ep_reward > highscore:
                        highscore = ep_reward
                    elapsed_time = time.time() - start_time
                    average_time_per_episode = elapsed_time / episode
                    output_str = f"Episode {episode}/{NUM_EPISODES}, Highscore: {highscore}, Reward: {ep_reward}, Epsilon: {round(self.epsilon, 3)}\n"
                    output_str += f"Total time: {round(elapsed_time, 1)}s, Episode time: {round(time.time() - ep_start, 1)}s, Average time: {round(average_time_per_episode, 1)}s\n"
                    output_str += f"No op: {action_counter[0]}, Left: {action_counter[3]}, Right: {action_counter[2]}, Total: {action_counter[0] + action_counter[2] + action_counter[3]}\n"
                    output_str += f"Total No op: {total_action_counter[0]}, Total Left: {total_action_counter[3]}, Total Right: {total_action_counter[2]}, memory size: {len(self.buffer)}\n\n"
                    write_to_file(output_str)
                    print(output_str, end='')  # print the output to the console

            self.epsilon = EPSILON_START - (episode/(NUM_EPISODES*(1/EPSILON_START)))
            if episode % SAVE_FREQ == 0:
                print(f"Model weights saved at episode {episode}")
                self.q_network.save_weights(f"{SAVE_PATH}model_{episode}.h5")
                with open(f"{SAVE_PATH}rewards.txt", "w") as f:
                    f.write(str(all_rewards))

if __name__ == "__main__":
    agent = RainbowAgent(TRAIN=True)
    agent.train()

