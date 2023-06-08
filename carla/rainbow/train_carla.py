from env_carla import CarEnv
from agent_carla import RainbowAgent
import time
import tensorflow as tf
import os
from datetime import datetime
import numpy as np



NUM_ACTIONS = 4
ACTIONS = [0, 1, 2, 3]
GAMMA = 0.99
ALPHA = 0.70
BETA = 0.4
BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 80_000
N_STEPS = 3
BATCH_SIZE = 2560
TRAINING_FREQ = 320
SHOW_FRAME = 100
TARGET_UPDATE_FREQ = 25_000
TARGET_UPDATE_RATE = 40
INITIAL_LEARNING_RATE = 0.001 #0.00025 #0.002
LEARNING_RATE_DECAY = 0.05   #0.05 for 5_000, 0.005 for 20_000
MIN_LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_DECAY = 0.04          #0.04 for 5_000, 0.01 for 20_000
SEED=42
VIDEO_LOCATION="F:/Coding/recordings/"
SAVE_PATH = "F:/Coding/carla/test/"
NUM_EPISODES = 1_000
SAVE_FREQ = 10
EMAIL_FREQUENCY = 1_000
RENDER = False
PRETRAINED = False
TRAINING = True
MODEL = ""
LOGGING = True
# INPUT_SHAPE = (None, 100, 130, 40, 40)
INPUT_SHAPE = (None, 4,)

def read_file(self, path):
        try:
            with open(path, "r") as f:
                return f.read()
        except:
            return ""
        
def estimate_score(self, total_episodes, scores):
        x = np.arange(len(scores))
        coefficients = np.polyfit(x, scores, 1)
        fitted_line = np.poly1d(coefficients)
        remaining_episodes = np.arange(len(scores), total_episodes)
        predicted_scores = fitted_line(remaining_episodes)
        return round(np.mean(predicted_scores[-100:]),1)

def estimate_remaining_time(self, total_episodes, times):
    x = np.arange(len(times))
    coefficients = np.polyfit(x, times, 1)
    fitted_line = np.poly1d(coefficients)
    remaining_episodes = np.arange(len(times), total_episodes)
    predicted_times = fitted_line(remaining_episodes)
    remaining_time = np.sum(predicted_times)
    return remaining_time

def send_email_notification(self, all_rewards, output_str):
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

            email_subject = self.save_path
            email_body = f"""{output_str}\n
    Average per 100: {average_per_100}\n"""
            self.write_to_file(email_body)
            # send_email(email_subject, f"{email_body}\nAll Rewards: {all_rewards}")
            print("\n")
        except Exception as e:
            print("email error", e)
            pass
        
def write_to_file(output_str, file_name="output.txt"):
    if TRAINING:
        try:
            with open(f"{SAVE_PATH}{file_name}", "a") as f:
                f.write(output_str)
        except:
            print("error writing to file")
            pass

def train():
    for episode in range(1000_000):
        steps = 0
        done = False
        env.video(episode)
        state = env.reset()
        episode_reward = 0
        # for _ in range(4):
            # agent.frame_buffer.append(state)
        # state = np.stack(agent.frame_buffer, axis=-1)
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            agent.frame_buffer.append(next_state)
            # next_state = np.stack(agent.frame_buffer, axis=-1)
            steps += 1
            agent.remember(state, action, reward, next_state, done)
            if steps % TRAINING_FREQ == 0 and TRAINING:
                learn_time, loss = agent.learn()
            if steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                write_to_file("updating target network\n")
        agent.epsilon = agent.decay(EPSILON_START, episode, EPSILON_DECAY)

        print(f"Episode: {episode}, Reward: {episode_reward}, Steps: {steps}, Epsilon: {agent.epsilon}")
        write_to_file(f"Episode: {episode}, Reward: {episode_reward}, Steps: {steps}, Epsilon: {agent.epsilon}\n")

        if (episode % SAVE_FREQ == 0 or episode == 10) and TRAINING:
            output_str = f"Model weights saved at episode {episode}\n"
            agent.q_network.save_weights(f"{SAVE_PATH}models/model_{episode}.h5")
            write_to_file(f"{output_str}\n")
            print(output_str)
if __name__ == "__main__":
    # tf.config.optimizer.set_jit(True)
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        os.makedirs(f"{SAVE_PATH}models")
    
    params = f"""Hyperparameters:
    Training Frequency: {TRAINING_FREQ}
    Target Update Frequency: {TARGET_UPDATE_FREQ}
    Initial Learning Rate: {INITIAL_LEARNING_RATE}
    Learning Rate Decay: {LEARNING_RATE_DECAY}
    Min Learning Rate: {MIN_LEARNING_RATE}
    Epsilon Start: {EPSILON_START}
    Epsilon Decay: {EPSILON_DECAY}
    Gamma: {GAMMA}
    Alpha: {ALPHA}
    Beta: {BETA}
    Buffer Size: {BUFFER_SIZE}
    Min Buffer Size: {MIN_BUFFER_SIZE}
    N Steps: {N_STEPS}
    Batch Size: {BATCH_SIZE}
    Input Shape: {INPUT_SHAPE}
    Pretrained: {PRETRAINED}
    Model: {MODEL}
    Training: {TRAINING}
    Actions: {ACTIONS}
    """
    write_to_file(params)
    print(params)
    # write_to_file(f"{read_file(path="")}", "code.py")
    env = CarEnv(render=True)
    agent = RainbowAgent(save_path=SAVE_PATH, num_actions=NUM_ACTIONS, gamma=GAMMA, alpha=ALPHA, beta=BETA, buffer_size=BUFFER_SIZE, min_buffer_size=MIN_BUFFER_SIZE, n_steps=N_STEPS, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ, input_shape=INPUT_SHAPE, pre_trained=PRETRAINED,num_episodes=NUM_EPISODES, learning_rate=INITIAL_LEARNING_RATE, epsilon=EPSILON_START, model_path=MODEL, training=TRAINING, actions=ACTIONS)
    human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_to_file(f"Starting training at {human_readable_time}\n\n")
    train()
    human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished training at {human_readable_time}")
    write_to_file(f"Finished training at {human_readable_time}\n")