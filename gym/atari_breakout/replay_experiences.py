
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np

with open("models_3_actions/experience_replay.pkl", 'rb') as f:
    memory = pickle.load(f)

if __name__ == "__main__":
    states, actions, rewards, next_states, dones = zip(*memory)
    print(len(states))
    states = None
    next_states = next_states[200_000:]
    # loop through states
    for state in next_states:
        plt.imshow(state, cmap='gray')
        fig = plt.show(block=False)
        plt.pause(0.001)  # Pause for a brief period
        plt.clf()
