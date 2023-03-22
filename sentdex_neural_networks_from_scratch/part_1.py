import numpy as np
import tensorflow as tf
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        
        
    def relu(self, z):
        
    
    def softmax(self, z):
        
    
    def forward(self, X):
    
    
    def loss(self, X, y):
        
    
    def train(self, X, y, learning_rate, epochs, batch_size):
        

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = x_train/255
X_test = x_test/255

num_classes = 10
# Convert the labels to one-hot encoding
y_train_onehot = np.eye(num_classes)[y_train] # shape (num_examples, output_size)

# Create the neural network
input_size = X_train.shape[1]
hidden_size = 128
output_size = num_classes
learning_rate = 0.1
epochs = 10
batch_size = 32
model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
model.train(X_train, y_train_onehot, learning_rate, epochs, batch_size)