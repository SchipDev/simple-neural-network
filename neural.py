import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


np.random.seed(0)

X = [[1, 2, 3, 2.5], 
     [2.0, 5.0, -1.0, 2.0], 
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

# Simple scalable neuron layer, takes the desired number of inputs and neurons as parameters
class Layer_Dense:
    
    # Generates weights and biases for the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Creates an out for the layer to pass on to another neural layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        


# Object desinged to execute the Rectified Linear Activation function on a neural layers output
class Activation_RectLin:
    
    # Executes the Rectified Linear Activation function by snipping vales <= 0
    def forward(self, input):
        self.output = np.maximum(0, input)      



# Demonstration
layer1 = Layer_Dense(2, 5)
activ_1 = Activation_RectLin()
layer1.forward(X)
activ_1.forward(layer1.output)   
print(activ_1.output) 


