import matplotlib.pyplot as plt

def plot_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # for line in lines[-1:]:
    data = lines[-1].strip().split(", ")
    print(len(data))
    print(data[:-100])
    data = list(map(int, data))
    plt.scatter([ _ for _ in range(len(data))], data, s=1)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Plot of Data")
    plt.show()

# Example usage:
plot_data_from_file("rainbow_models/sample.txt")
