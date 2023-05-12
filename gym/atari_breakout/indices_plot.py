import matplotlib.pyplot as plt

def plot_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # for line in lines[-1:]:
    # print(lines[1])
    data = lines[10000].strip().split(",")
    # print(len(data))
    # print(data[:-100])
    data = list(map(int, data))
    plt.scatter([ _ for _ in range(len(data))], data, s=1)

    plt.xlabel("step")
    plt.ylabel("index")
    plt.title("index vs step")
    plt.show()

# Example usage:
plot_data_from_file("F:/Coding/breakout/prioritized_3_n_step_dueling/samples.txt")
