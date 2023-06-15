import gymnasium as gym
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# Define the actor model architecture

def create_actor_model():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(1024, activation="relu")(inputs)
    out = layers.Dense(1024, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * 2
    model = tf.keras.Model(inputs, outputs)
    return model

# Create the Pendulum-v1 environment
# problem = 'Pusher-v4'
# problem = 'Pendulum-v1'
problem = "Humanoid-v4"
env = gym.make(problem, render_mode='human')
num_episodes = 1000
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# Load the saved weights into the actor and critic models
actor_model = create_actor_model()
# actor_model.load_weights('pendulum_actor.h5')
actor_model.load_weights(f'{problem}/actor_3000.h5')

# Play episodes using the loaded actor-critic models


for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    state = state[0]
    step = 0
    while not done:
        # print(state)
        # print(state.shape)
        # Take an action using the actor model
        # state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        state = state.reshape((1, num_states))
        action = actor_model(state).numpy()[0]
        # print(action)
        # Perform the action in the environment
        next_state, reward, done, _, info = env.step(action)
        # Update the total reward
        total_reward += reward

        # Update the state for the next iteration
        state = next_state
        step += 1
        if step % 300 == 0:
            done = True

    print(f"Episode {episode+1} - Total Reward: {total_reward}")

# Close the environment
env.close()
