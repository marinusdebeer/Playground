import numpy as np
beta = 0.4
# probs = np.array([0.1, 0.2, 0.3, 0.4])
# weights = (100_000 * probs) ** (-beta)
# # weights /= (len(self.memory) * probs.min()) ** (-beta)
# print(weights)
# weights /= weights.max()
# print(weights)

priorities = np.array([1.1, 0.2, 3.3, 0.4])
weights = (priorities ** (-beta)) / np.max(priorities)
print(weights)