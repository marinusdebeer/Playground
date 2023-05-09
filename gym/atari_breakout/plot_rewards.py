import matplotlib.pyplot as plt
import numpy as np
import ast
# with open("rainbow_models/rewards.txt", "r") as f:
# with open("F:/Coding/breakout/rewards.txt", "r") as f:
with open("F:/Coding/breakout/noNormalizationWithSteps_2/rewards.txt", "r") as f:
# with open("F:/Coding/breakout/normalizedModelSteps/rewards.txt", "r") as f:
# with open("models_3_actions/rewards.txt", "r") as f:
# with open("rainbow_models/good_models/rewards_1.txt", "r") as f:
    data_str = f.read()
    data = ast.literal_eval(data_str)

fig, ax = plt.subplots()
data_array = np.array(data)
# print(np.average(data_array[5000:]))
total = 7500
interval = 100
# reshaped_arr = data_array[-2_000:].reshape(40, 50)
print(data_array[-total:].mean())
reshaped_arr = data_array[-total:].reshape(total//interval, interval)

averaged_arr = reshaped_arr.mean(axis=1)
print(averaged_arr)
ax.plot(averaged_arr)
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
ax.set_title('Rewards over time')
plt.show()
