import gymnasium as gym
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import random
from collections import deque

class ActorCritic:
	def __init__(self, env):
		self.env  = env

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau   = .125
		self.num_actions = env.action_space.shape[0]
		self.num_states  = env.observation_space.shape[0]

		self.memory = deque(maxlen=BUFFER_SIZE)
		self.actor_model = self.create_actor_model()
		self.target_actor_model = self.create_actor_model()

		self.critic_model = self.create_critic_model()
		self.target_critic_model = self.create_critic_model()

	def create_actor_model(self):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = layers.Input(shape=(self.num_states,))
		out = layers.Dense(256, activation="relu")(inputs)
		out = layers.Dense(256, activation="relu")(out)
		outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)

		# Our upper bound is 2.0 for Pendulum.
		outputs = outputs * 2
		model = tf.keras.Model(inputs, outputs)
		return model
	
	def create_critic_model(self):
		# State as input
		state_input = layers.Input(shape=(self.num_states))
		state_out = layers.Dense(16, activation="relu")(state_input)
		state_out = layers.Dense(32, activation="relu")(state_out)

		# Action as input
		action_input = layers.Input(shape=(self.num_actions))
		action_out = layers.Dense(32, activation="relu")(action_input)

		# Both are passed through seperate layer before concatenating
		concat = layers.Concatenate()([state_out, action_out])

		out = layers.Dense(256, activation="relu")(concat)
		out = layers.Dense(256, activation="relu")(out)
		outputs = layers.Dense(1)(out)

		# Outputs single value for give state-action
		model = tf.keras.Model([state_input, action_input], outputs)

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append([state, action, reward, next_state, done])

	def _train_actor(self, samples):
		# print("Training Actor")
		states, actions, rewards, next_states, dones = zip(*samples)
		states = np.array(states)
		# actions = np.array(actions)
		# rewards = np.array(rewards)
		# next_states = np.array(next_states)
		# dones = np.array(dones)
		with tf.GradientTape() as tape:
			predicted_actions = self.actor_model(states)
			critic_values = self.critic_model([states, predicted_actions])
			actor_loss = -tf.reduce_mean(critic_values)
		actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
		self.actor_model.optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

            
	def _train_critic(self, samples):
		# print("Training Critic")
		states, actions, rewards, next_states, dones = zip(*samples)
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)
		with tf.GradientTape() as tape:
			target_actions = self.target_actor_model(next_states).numpy()
			future_rewards = self.target_critic_model([next_states, target_actions]).numpy()

			rewards = rewards.reshape((rewards.shape[0], 1))
			rewards += self.gamma * future_rewards
			
			critic_values = self.critic_model([states, actions])
			critic_loss = tf.reduce_mean(tf.square(rewards - critic_values))
			
		critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
		self.critic_model.optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
		
	def train(self):
		if len(self.memory) < BATCH_SIZE:
			return
		# print("Training")
		samples = random.sample(self.memory, BATCH_SIZE)
		self._train_critic(samples)
		# self._train_actor(samples)
		# print("Done Training")

	def _update_model(self, target_model, model):
		weights = model.get_weights()
		target_weights = target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		target_model.set_weights(target_weights)

	def update_target(self):
		self._update_model(self.target_actor_model, self.actor_model)
		self._update_model(self.target_critic_model, self.critic_model)

	def act(self, state, episode):
		self.epsilon = INITIAL_EPSILON * (DECAY_RATE ** (episode/NUM_EPISODES))
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return self.actor_model(state.reshape(1, self.env.observation_space.shape[0])).numpy()[0]

NUM_EPISODES = 1_000
BATCH_SIZE = 2560
TRAINING_FREQ = 320
DECAY_RATE = 0.04
INITIAL_EPSILON = 1.0
SAVE_FREQ = 10
SAVE_PATH = "pendulum_models"
TRAINING = True
STEPS_PER_EPISODE = 10_000
BUFFER_SIZE = 100_000
def main():
	env = gym.make("Pendulum-v1")
	# env = gym.make("Pendulum-v1", render_mode='human')
	# env = gym.make("Pusher-v4", render_mode='human')
	actor_critic = ActorCritic(env)

	step = 0
	action = env.action_space.sample()
	print("Starting training...")
	for episode in range(NUM_EPISODES):
		done = False
		state = env.reset()
		state = state[0]
		ep_reward = 0
		while not done:
			# state = state.reshape((1, env.observation_space.shape[0]))
			action = actor_critic.act(state, episode)
			# action = action.reshape((1, env.action_space.shape[0]))[0]
			# print(action)
			next_state, reward, done, _, info = env.step(action)

			# next_state = next_state.reshape((1, env.observation_space.shape[0]))
			ep_reward += reward
			actor_critic.remember(state, action, reward, next_state, done)
			if step % TRAINING_FREQ == 0:
				actor_critic.train()

			state = next_state
			step += 1
			if step % STEPS_PER_EPISODE == 0:
				done = True
		if (episode % SAVE_FREQ == 0 or episode == 0) and TRAINING:
			print(f"Saving model at episode {episode}")
			actor_critic.critic_model.save_weights(f"{SAVE_PATH}/actor/model_{episode}.h5")
			actor_critic.actor_model.save_weights(f"{SAVE_PATH}/critic/model_{episode}.h5")
			print(f"episode: {episode}, reward: {round(ep_reward)}, epsilon: {round(actor_critic.epsilon, 6)}, steps: {step}")

if __name__ == "__main__":
	main()
