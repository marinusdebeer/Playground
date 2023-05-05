import matplotlib.pyplot as plt
import numpy as np
import ast
# create some sample data
with open("rainbow_models/rewards.txt", "r") as f:
    data_str = f.read()
    data = ast.literal_eval(data_str)

# create a figure and axis object
fig, ax = plt.subplots()
data_array = np.array(data)
# print(np.average(data_array[5000:]))

reshaped_arr = data_array[-5500:].reshape(55, 100)
averaged_arr = reshaped_arr.mean(axis=1)
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
