from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

import matplotlib.pyplot as plt

# Define class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess(x, y, limit):
    x = x[:limit].reshape(limit, 1, 28, 28)
    x = x.astype("float32") / 255
    y = y[:limit]
    arr_Y = np.zeros((y.size, 10))  # Now all the elements in the matrix are zeroes
    arr_Y[np.arange(y.size), y] = 1
    y = arr_Y.reshape(limit, 10, 1)
    return x, y


class Layer:
    def __init__(self):
        self.input = None
        self.output = None  
        self.trainable = True  # Default to trainable

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass
    def set_trainable(self, trainable):
        self.trainable = trainable


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        if not self.trainable:
            return np.dot(self.weights.T, output_gradient)
        
        weights_gradient = np.dot(output_gradient, self.input.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
        
    def save_weights(self, filename):
        np.savez(filename, weights=self.weights, bias=self.bias)
    
    def load_weights(self, filename):
        data = np.load(filename)
        self.weights = data['weights']
        self.bias = data['bias']

class Activation(Layer):
    def __init__(self, activation, activation_gradient):
        super().__init__()

        self.activation = activation
        self.activation_gradient = activation_gradient
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_gradient(self.input))

class Convolutional(Layer):
  def __init__(self, input_shape, kernel_size, num_filters):
    super().__init__()
    self.input_shape = input_shape
    self.input_channels = self.input_shape[0]
    self.input_height = self.input_shape[1]
    self.input_width = self.input_shape[2]
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.output_shape = (self.num_filters, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
    self.kernels_shape = (self.num_filters, self.input_channels, kernel_size, kernel_size)
    self.kernels = np.random.randn(*self.kernels_shape) / (kernel_size * kernel_size)
    self.biases = np.random.randn(*self.output_shape)

  def image_region(self, image):
    input_height, input_width = image.shape[1:] 
    for i in range(self.input_channels):
      for j in range(input_height - self.kernel_size + 1):
        for k in range(input_width - self.kernel_size + 1):
          image_patch = image[i, j:(j+self.kernel_size), k:(k+self.kernel_size)]
          yield image_patch, i, j, k

  def forward(self, input_data):
    self.input_data = input_data
    output = np.zeros(self.output_shape)
    for image_patch, i, j, k in self.image_region(input_data):
      for n in range(self.num_filters):
        output[n, j, k] += np.sum(image_patch * self.kernels[n, i]) + self.biases[n, j, k]
    return output
  
  def backward(self, output_gradient, learning_rate):
    if not self.trainable:
      return np.zeros(self.input_shape)
    kernels_gradient = np.zeros(self.kernels_shape) 

    input_gradient = np.zeros(self.input_shape)  

    # Calculate gradients for kernels
    for image_patch, i, j, k in self.image_region(self.input_data):
        for n in range(self.num_filters):
            kernels_gradient[n, i] += image_patch * output_gradient[n, j, k] 

    # Calculate input gradient
    for image_patch, i, j, k in self.image_region(self.input_data):
        for n in range(self.num_filters):
            flipped_kernel = np.flipud(np.fliplr(self.kernels[n, i]))
            input_gradient[i, j:(j+self.kernel_size), k:(k+self.kernel_size)] += output_gradient[n, j, k] * flipped_kernel



    self.kernels -= learning_rate * kernels_gradient
    self.biases -= learning_rate * output_gradient

    return input_gradient
  
  def save_weights(self, filename):
      np.savez(filename, kernels=self.kernels, biases=self.biases)

  def load_weights(self, filename):
      data = np.load(filename)
      self.kernels = data['kernels']
      self.biases = data['biases']

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        
        return np.reshape(output_gradient, self.input_shape)
    
class Softmax(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_gradient(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(x, 0.33)

def relu_gradient(x):
    return x > 0.33

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_gradient(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            y = y.reshape(-1,1)
            output = predict(network, x)

            error += loss(y, output)

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

def predict_final(network, x):
    x = x.reshape(1, 28, 28)
    x = x.astype("float32") / 255
    output = predict(network, x)
    return output