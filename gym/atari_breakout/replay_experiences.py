
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np

with open("models_3_actions/experience_replay.pkl", 'rb') as f:
    memory = pickle.load(f)

if __name__ == "__main__":
    states, _, _, _, _ = zip(*memory)
    print(len(states))
    states = states[240_000:]
    
    print(states[0].shape)

    # loop through states
    for state in states:
        plt.imshow(state[:,:,0], cmap='gray')
        fig = plt.show(block=False)
        plt.pause(0.001)  # Pause for a brief period
        plt.clf()
