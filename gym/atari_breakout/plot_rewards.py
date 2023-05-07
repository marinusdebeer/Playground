import matplotlib.pyplot as plt
import numpy as np
import ast
# create some sample data
with open("rainbow_models/rewards.txt", "r") as f:
# with open("rainbow_models/good_models/rewards_1.txt", "r") as f:
    data_str = f.read()
    data = ast.literal_eval(data_str)

# create a figure and axis object
fig, ax = plt.subplots()
data_array = np.array(data)
# print(np.average(data_array[5000:]))

# reshaped_arr = data_array[-2_000:].reshape(40, 50)
reshaped_arr = data_array[-2000:].reshape(20, 100)

averaged_arr = reshaped_arr.mean(axis=1)

cumulative_distances = np.cumsum(averaged_arr)
elapsed_times = np.arange(len(averaged_arr)) * 100
def distance_function(t):
    return np.interp(t, elapsed_times, cumulative_distances)
t = 5000
d = distance_function(t)
print("Distance traveled at t=", t, "units of time:", d)
# v = (np.diff(averaged_arr)/200).mean()
# print(v)
# a = (np.diff(np.diff(averaged_arr)/200)/200).mean()
# print(a)
# distance = v*1400 + 0.5*a*1400**2
# print(distance)

np.savetxt('my_array.csv', averaged_arr, delimiter=',')
# Print the result
print(averaged_arr)
# plot the data as a line graph
ax.plot(averaged_arr)

# set the x-axis label
ax.set_xlabel('Episodes')

# set the y-axis label
ax.set_ylabel('Rewards')

# set the title of the plot
ax.set_title('Rewards over time')

# show the plot
plt.show()
