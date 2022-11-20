import math

def sigmoid(x):
    """
    activation function for neuron
    """
    return 1.0 / (1 + math.exp(-x))

def activate(inputs, weights):
    """
    perform net input
    perform activation
    """
    net = 0
    for x, w in zip(inputs, weights):
        net += x * w
    
    return sigmoid(net)

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print(output)

    