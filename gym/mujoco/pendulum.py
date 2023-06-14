import gymnasium as gym
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

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

		self.memory = deque(maxlen=BUFFER_SIZE)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		h1 = Dense(24, activation='relu')(state_input)
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
		
		model = Model(inputs=state_input, outputs=output)
		adam  = Adam(learning_rate=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model
	
	def create_critic_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=self.env.action_space.shape)
		action_h1 = Dense(48)(action_input)
		
		merged = tf.keras.layers.concatenate([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model = Model(inputs=[state_input, action_input], outputs=output)
		
		adam = Adam(learning_rate=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		print("Training Actor")
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			with tf.GradientTape() as tape:
				predicted_action = self.actor_model(cur_state)
				actor_loss = -self.critic_model([cur_state, predicted_action])
			actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
			self.actor_model.optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
            
	def _train_critic(self, samples):
		print("Training Critic")
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model(new_state)
				future_reward = self.target_critic_model([new_state, target_action])
				reward += self.gamma * future_reward
			with tf.GradientTape() as tape:
				critic_value = self.critic_model([cur_state, action])
				critic_loss = tf.math.reduce_mean(tf.square(reward - critic_value))
			critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
			self.critic_model.optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
		
	def train(self):
		if len(self.memory) < BATCH_SIZE:
			return
		print("Training")
		samples = random.sample(self.memory, BATCH_SIZE)
		self._train_critic(samples)
		self._train_actor(samples)
		print("Done Training")

	def _update_model(self, target_model, model):
		weights = model.get_weights()
		target_weights = target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		target_model.set_weights(target_weights)

	def update_target(self):
		self._update_model(self.target_actor_model, self.actor_model)
		self._update_model(self.target_critic_model, self.critic_model)

	def act(self, cur_state, episode):
		self.epsilon = INITIAL_EPSILON * (DECAY_RATE ** (episode/NUM_EPISODES))
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon or True:
			return self.env.action_space.sample()
		return self.actor_model.predict(cur_state)

NUM_EPISODES = 10_000
BATCH_SIZE = 2560
TRAINING_FREQ = 320
DECAY_RATE = 0.04
INITIAL_EPSILON = 1.0
SAVE_FREQ = 10
SAVE_PATH = "pendulum_models"
TRAINING = True
STEPS_PER_EPISODE = 20_000
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
		cur_state = env.reset()
		cur_state = cur_state[0]
		ep_reward = 0
		while not done:
			cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
			action = actor_critic.act(cur_state, episode)
			action = action.reshape((1, env.action_space.shape[0]))[0]
			# print(action)
			new_state, reward, done, _, info = env.step(action)

			new_state = new_state.reshape((1, env.observation_space.shape[0]))
			ep_reward += reward
			actor_critic.remember(cur_state, action, reward, new_state, done)
			if step % TRAINING_FREQ == 0:
				actor_critic.train()

			cur_state = new_state
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
