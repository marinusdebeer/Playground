# https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time
# from send_email import send_email
from dotenv import load_dotenv
load_dotenv()
    
class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()

    def create_model(self):
        state_input = Input (shape=self.state_dim)
        h1 = Dense (24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense (24, activation='relu')(h2)
        output = Dense(self.action_dim, activation='relu')(h3)
        model = Model(state_input, output)
        adam = Adam(learning_rate=0.001)
        model.compile(loss="mse", optimizer=adam)


        # state_input = layers.Input(shape=(self.state_dim,))
        # dense_layer1 = layers.Dense(400, activation='relu')(state_input)
        # dense_layer2 = layers.Dense(300, activation='relu')(dense_layer1)
        # output_layer = layers.Dense(self.action_dim, activation='tanh')(dense_layer2)
        # output_layer *= self.action_bound

        # model = tf.keras.Model(state_input, output_layer)
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model
    
class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()

    def create_model(self):
        state_input = Input (shape=self.state_dim)
        state_h1 = Dense (24, activation='relu')(state_input)
        state_h2 = Dense (48)(state_h1)

        action_input = Input(shape=self.action_dim)
        action_h1 = Dense (48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model([state_input,action_input], output)

        adam = Adam(learning_rate=0.001)
        model.compile(loss="mse", optimizer=adam)

        # state_input = layers.Input(shape=(self.state_dim,))
        # action_input = layers.Input(shape=(self.action_dim,))
        # dense_layer1 = layers.Dense(400, activation='relu')(state_input)
        # dense_layer2 = layers.Dense(300, activation='relu')(dense_layer1)
        # output_layer = layers.Dense(1)(dense_layer2)

        # model = tf.keras.Model([state_input, action_input], output_layer)
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

        return model
    
class RainbowAgent:
    def __init__(self, state_size, action_bound, save_path, num_actions, learning_rate, epsilon, target_update_freq, beta, buffer_size, num_episodes, alpha, n_steps, model_path, pre_trained, training, gamma, min_buffer_size, input_shape, batch_size, actions):
        self.state_size = state_size
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
        
        self.save_path = save_path
        self.frame_buffer = deque(maxlen=4)
        self.buffer = deque(maxlen=buffer_size)
        
        self.fig_frame = None

        self.action_bound = action_bound

        self.actor = Actor(self.state_size, self.num_actions, self.action_bound)
        self.critic = Critic(self.state_size, self.num_actions)
        self.target_actor = Actor(self.state_size, self.num_actions, self.action_bound)
        self.target_critic = Critic(self.state_size, self.num_actions)

        self.update_target_network(tau=1)  # Make the target networks start off identical to the original networks
    
    def decay(self, initial, episode, decay_rate):
        return initial * (decay_rate ** (episode/self.num_episodes))

    def update_target_network(self, tau):
        if self.training:
            print("updating target network")
            actor_weights = self.actor.model.get_weights()
            critic_weights = self.critic.model.get_weights()
            actor_target_weights = self.target_actor.model.get_weights()
            critic_target_weights = self.target_critic.model.get_weights()

            for i in range(len(actor_weights)):
                actor_target_weights[i] = tau*actor_weights[i] + (1-tau)*actor_target_weights[i]

            for i in range(len(critic_weights)):
                critic_target_weights[i] = tau*critic_weights[i] + (1-tau)*critic_target_weights[i]

            self.target_actor.model.set_weights(actor_target_weights)
            self.target_critic.model.set_weights(critic_target_weights)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    

    def train_actor (self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={self.critic_state_input: cur_state, self.critic_action_input: predicted_action})[0]
            self.sess.run(self.optimize, feed_dict={self.actor_state_input: cur_state, self.actor_critic_grad: grads})

    def train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0] 
                reward += self.gamma * future_reward
                self.critic_model.fit([cur_state, action], reward, verbose=0)

    def learn(self):
        if len(self.buffer) < self.min_buffer_size:
            return 0, 0
        start = time.time()
        minibatch = random.sample(self.buffer, self.batch_size)

        self.train_actor(minibatch)
        self.train_critic(minibatch)

        # states, actions, rewards, next_states, dones = zip(*minibatch)
        # states = np.array(states, dtype=np.float32)
        # actions = np.array(actions, dtype=np.float32)
        # rewards = np.array(rewards, dtype=np.float32)
        # next_states = np.array(next_states, dtype=np.float32)
        # # dones = np.array(dones, dtype=np.float32)
        # # print(actions[0])
        # act_predict = self.target_actor.model.predict(next_states, verbose=0)
        # target_q = self.target_critic.model.predict([next_states, act_predict], verbose=0)
        # self.critic.model.train_on_batch([states, actions], rewards + target_q)
        # with tf.GradientTape(persistent=True) as tape:
        #     actions_predict = self.actor.model(states)
        #     critic_value = self.critic.model([states, actions_predict])
        #     actor_loss = -tf.math.reduce_mean(critic_value)
        
        # print(actor_loss.numpy())
        # actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        # self.actor.model.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

        end = time.time()
        return round(end - start, 2), 0

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(low=-2.0, high=2.0, size=self.num_actions)
        else:
            return self.actor.model.predict(state, verbose=None)[0]