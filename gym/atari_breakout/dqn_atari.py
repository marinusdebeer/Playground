import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

# Define hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQUENCY = 10000
MEMORY_SIZE = 1000000
PREPROCESS_SIZE = (84, 84)
NUM_FRAMES = 4
# MAX_STEPS = 50_000_000
MAX_STEPS = 5_000

# Define the preprocessing function


def preprocess(observation):
    print(observation.shape)
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation, PREPROCESS_SIZE)
    observation = observation / 255.0
    return observation.numpy()

# Define the Deep Q-Network model


def build_model(input_shape, num_actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=4, activation='relu',
               input_shape=input_shape),
        Conv2D(64, (4, 4), strides=2, activation='relu'),
        Conv2D(64, (3, 3), strides=1, activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_actions)
    ])
    model.compile(loss='mse', optimizer=RMSprop(lr=LEARNING_RATE))
    return model

# Define the replay buffer


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Define the main function


def main():
    print("STARTING")
    # Initialize the environment and the replay buffer
    env = gym.make('ALE/Breakout-v5', render_mode="human")
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    # Initialize the Deep Q-Network models
    input_shape = (PREPROCESS_SIZE[0], PREPROCESS_SIZE[1], NUM_FRAMES)
    num_actions = env.action_space.n
    model = build_model(input_shape, num_actions)
    target_model = build_model(input_shape, num_actions)
    target_model.set_weights(model.get_weights())

    # Initialize the state and the score
    state = env.reset()
    state = np.stack([preprocess(state)] * NUM_FRAMES, axis=2)
    score = 0

    # Main loop
    for step in range(MAX_STEPS):
        # Select an action using epsilon-greedy exploration
        epsilon = max(EPSILON_END, EPSILON_START -
                      (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY)
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(q_values[0])

        # Execute the action and store the experience in the
        # replay buffer

        next_state, reward, done, _ = env.step(action)
        next_state = np.append(state[:, :, 1:], np.expand_dims(
            preprocess(next_state), axis=2), axis=2)
        replay_buffer.add((state, action, reward, next_state, done))

        # Update the state and the score
        state = next_state
        score += reward

        # Train the model using experience replay
        if len(replay_buffer.buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                BATCH_SIZE)

            # Compute the target Q-values using the Double DQN algorithm
            q_values_next = target_model.predict(next_states)
            best_actions = np.argmax(model.predict(next_states), axis=1)
            q_values_next_target = q_values_next[np.arange(
                BATCH_SIZE), best_actions]
            targets = rewards + GAMMA * q_values_next_target * (1 - dones)

            # Compute the predicted Q-values and the loss
            mask = tf.one_hot(actions, num_actions)
            with tf.GradientTape() as tape:
                q_values = model(states)
                predicted = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
                loss = tf.reduce_mean(tf.square(targets - predicted))

            # Compute the gradients and update the model
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

        # Update the target model
        if step % TARGET_UPDATE_FREQUENCY == 0:
            target_model.set_weights(model.get_weights())

        # Print the score and the epsilon value
        if done:
            print(f'step={step}, score={score}, epsilon={epsilon:.3f}')
            score = 0

        # Reset the environment if the game is over
        if done or step == MAX_STEPS - 1:
            state = env.reset()
            state = np.stack([preprocess(state)] * NUM_FRAMES, axis=2)
    # Save the model
    model.save_weights('model.h5')
main()