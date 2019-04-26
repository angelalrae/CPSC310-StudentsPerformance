
import math
import random

# -------------------------------------------------------------
# 1: Simple Perceptron
# -------------------------------------------------------------

# PERCEPTRON: approximates a single neuron with n binary inputs
def step_function(x):
    return 1 if x >= 0 else 0

def dot(v, w):
    return sum(vi * wi for vi, wi in zip(v, w))

def perceptron_output(weights, bias, x):
    '''
    returns 1 if the perceptron 'fires', 0 if not
    '''
    calculation = dot(weights, x) + bias
    return step_function(calculation)

weights = [2, 2]
bias = -1
x1 = [1, 1]
x2 = [1, 0]
x3 = [0, 0]
x4 = [0, 1]

print('[1, 1]: %r' % perceptron_output(weights, bias, x1))
print('[1, 0]: %r' % perceptron_output(weights, bias, x2))
print('[0, 0]: %r' % perceptron_output(weights, bias, x3))
print('[0, 1]: %r' % perceptron_output(weights, bias, x4))

# -------------------------------------------------------------
# 2: Feed-forward algorithm
# -------------------------------------------------------------

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    '''
    takes in a neural network (represented as a list of lists
    of lists of weights) and returns the output from forward-
    propagating the input
    '''
    outputs = []
    # process one layer at a time
    for layer in neural_network:                            # add bias input
        input_with_bias = input_vector + [1]                # compute the output
        output = [neuron_output(neuron, input_with_bias)    # for each neuron
            for neuron in layer]                            # and remember it
        outputs.append(output)
        input_vector = output
    
    return outputs

# The network is made up of a 'hidden layer' and a 'output layer,
# the hidden layer contains an and and an or neuron.
xor_network =  [[20, 20, -30], [20, 20, -10]], [[-60, 60, -30]]

for x in [0, 1]:
    for y in [0, 1]:
        # feed_forward produces the outputs of every neuron
        # feed_forward[-1] is the outputs of the output-layer neurons
        print(x, y, feed_forward(xor_network, [x, y]))




# -------------------------------------------------------------
# 3: Backpropagation
# -------------------------------------------------------------

# a. run feed-_forward on an input vector to produce the outputs
#       of all the neurons in the network
# b. this results in an error for each output neuron - the difference
#       between its output and its target
# c. compute the gradient of this error as a function of the neuron's 
#       weights, and adjust its weights in the direction that most 
#       decreases the error
# d. "Propagate" these output errors backward to infer errors for 
#       the hidden layer
# e. compute the gradients of these errors and adjust the hidden 
#       layer's weights in the same manner
def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target) 
                    for output, target in zip(outputs, targets)]
    
    # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output
        
    # back_propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                    dot(output_deltas, [n[i] for n in outputs])
                    # this says actually "output layers, change if unsatisfactory"
                    for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input


zero_digit =    [1,1,1,1,1,
                 1,0,0,0,1,
                 1,0,0,0,1,
                 1,0,0,0,1,
                 1,1,1,1,1]

targets = [[1 if i == j else 0 for i in range(10)] for j in range(10)]

random.seed(0)
input_size = 25
num_hidden = 5
output_size = 10

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for _ in range(input_size + 1)]
                for _ in range(num_hidden)]

# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for _ in range(num_hidden + 1)] 
                for _ in range(output_size)]

# the network starts out with random weights
network = [hidden_layer, output_layer]


def predict(network, )
    # and we can train it using the backpropagation algorithm
    # 10,000 iterations will (hopefully) converge:
    for _ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)