# https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from collections import deque
import random
import time
# from send_email import send_email
from dotenv import load_dotenv
load_dotenv()

class DQN(tf.keras.Model):
    def __init__(self, num_actions, input_shape):
        super(DQN, self).__init__()
        self.dense1 = Dense(32, activation='relu', input_shape=input_shape)
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(64, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense4 = Dense(32, activation='relu')
        self.output_layer = Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.flatten(x)
        x = self.dense4(x)
        q_values = self.output_layer(x)
        return q_values
class RainbowAgent:
    def __init__(self, save_path, num_actions, learning_rate, epsilon, target_update_freq, beta, buffer_size, num_episodes, alpha, n_steps, model_path, pre_trained, training, gamma, min_buffer_size, input_shape, batch_size, actions):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.beta = beta
        self.beta_increment = None
        self.buffer_size = buffer_size
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.n_steps = n_steps
        self.model_path = model_path
        self.pre_trained = pre_trained
        self.training = training
        self.gamma = gamma
        self.min_buffer_size = min_buffer_size
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.actions = actions
        

        self.save_path = save_path
        self.frame_buffer = deque(maxlen=4)
        self.buffer = deque(maxlen=buffer_size)
        
        self.fig_frame = None
        self.q_network = DQN(self.num_actions, input_shape=self.input_shape)
        self.target_network = DQN(self.num_actions, input_shape=self.input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if pre_trained:
            dummy_input = np.zeros((1, 84, 84, 4))
            self.q_network(dummy_input)
            self.q_network.load_weights(self.model_path)
            self.target_network(dummy_input)

        self.update_target_network()
    def decay(self, initial, episode, decay_rate):
        return initial * (decay_rate ** (episode/self.num_episodes))

    def update_target_network(self):
        if self.training:
            print("updating target network")
            self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    # Use tf.function to speed up the computation graph
    @tf.function
    def compute_loss_and_gradients(self, states, actions, rewards, next_states, dones, weights=None):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)

            next_q_values = self.target_network(next_states)
            next_q_values_online = self.q_network(next_states)

            next_actions = tf.argmax(next_q_values_online, axis=1)
            next_q_values = tf.reduce_sum(next_q_values * tf.one_hot(next_actions, self.num_actions), axis=1)

            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
            target_q_values = tf.stop_gradient(target_q_values)
            td_errors = target_q_values - q_values
            loss = tf.keras.losses.MSE(target_q_values, q_values)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        return loss, gradients, td_errors

    def learn(self):
        if len(self.buffer) < self.min_buffer_size:
            return 0, 0
        # print("learning")
        start = time.time()
        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        loss, gradients, td_errors = self.compute_loss_and_gradients(states, actions, rewards, next_states, dones, None)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        end = time.time()
        return round(end - start, 2), loss.numpy()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon and self.training:
            action_probs = [0.15, 0.64, 0.15, 0.06]  # Probabilities for each action
            return np.random.choice(len(action_probs), p=action_probs)
        else:
            q_values = self.q_network(np.array([state], dtype=np.float32))
            action = self.actions[np.argmax(q_values.numpy())]
        return action