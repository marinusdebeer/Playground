import numpy as np
import gymnasium as gym
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from collections import deque
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm

class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='relu', input_shape=(84, 84, 4))
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
    def __init__(self):
        self.num_actions = 2
        self.memory = deque(maxlen=100_000)
        self.frame_buffer = deque(maxlen=4)
        self.epsilon = 0.997
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.00025

        self.num_episodes = 1000
        self.target_update_freq_steps = 5_000
        self.batch_size = 32
        self.training_frequency = 4
        self.render = True

        self.model = DQN(self.num_actions)
        self.target_model = DQN(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.update_target_network()

    def preprocess_state(self, state):
      gray = rgb2gray(state[0])
      resized = resize(gray, (84, 84), mode='constant')
      processed_observe = np.uint8(resized * 255)
      return processed_observe

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([3, 4])
            # return np.random.choice(self.num_actions)
        q_values = self.model(np.array([state], dtype=np.float32))
        action = np.argmax(q_values.numpy()[0]) + 3
        # print("......................................action: ", action)
        return action

    def update_target_network(self):
        print("updating target network")
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, step):
        if len(self.memory) < self.batch_size or step % self.training_frequency != 0:
            return
        # print("training")
        minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        # dones = np.array(dones, dtype=np.bool)
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
        step_count = 0
        if self.render:
          env = gym.make('BreakoutDeterministic-v4', render_mode="human", full_action_space=True)
        else:
          env = gym.make('BreakoutDeterministic-v4', full_action_space=True)
        for episode in tqdm(range(1, self.num_episodes + 1), ascii=True, unit='episodes'):
            fire = True
            action = 16
            lives = 5
            state = env.reset()
            for _ in range(4):
                self.frame_buffer.append(self.preprocess_state(state))
            state = np.stack(self.frame_buffer, axis=-1)
            done = False
            episode_reward = 0
            
            while not done:
                # env.render()
                if not fire:
                  action = self.choose_action(state)
                next_state, reward, done, _, info = env.step(action)
                if not fire:
                  self.frame_buffer.append(self.preprocess_state(next_state))
                  next_state = np.stack(self.frame_buffer, axis=-1)
                  self.remember(state, action-2, reward, next_state, done)
                  state = next_state
                  episode_reward += reward
                  step_count += 1
                  self.replay(step_count)

                if(info["lives"] < lives):
                  lives = info["lives"]
                  fire = True
                  action = 16
                else:
                  fire = False
               
                if done:
                    print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}")
                if step_count % self.target_update_freq_steps == 0:
                  self.update_target_network()

            if self.epsilon > self.epsilon_min:
                self.epsilon = pow(self.epsilon_decay, episode)

            if episode % 10 == 0 or episode == 0:
                self.model.save_weights(f"models/model_weights_episode_{episode}.h5")
                print(f"Model weights saved at episode {episode}")

if __name__ == "__main__":
    agent = DQNAgent()
    # dummy_input = np.zeros((1, 84, 84, 4))
    # agent.model(dummy_input)
    # agent.model.load_weights("models/model_weights_episode_730.h5")
    agent.train()