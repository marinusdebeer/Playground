import numpy as np
from rl_breakout import DQNAgent
agent = DQNAgent(2)
dummy_input = np.zeros((1, 84, 84, 4))
_ = agent.model(dummy_input)
agent.model.load_weights("models/model_weights_episode_560.h5")
agent.model.summary()