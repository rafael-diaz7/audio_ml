import numpy as np


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        """
        teafnaefnaso
        :param num_inputs: number of input nodes as int
        :param num_hidden: list of ints, where len is = num of hidden, item is num of nodes in hidden
        :param num_outputs: number of output nodes as int
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # (3, 3, 5, 2)
        # initiate random weights for all connections
        self.weights = []
        for i in range(len(layers) - 1):
            # going to i - 1 because weight matrices are between layers
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        """

        :param inputs:
        :return:
        """
        activations = inputs
        # go through each layer
        for weight in self.weights:
            #  calculate net inputs
            net_inputs = np.dot(activations, weight)

            # calculate the activations
            activations = self._sigmoid(net_inputs)
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":

    # create an MLP
    mlp = MLP()
    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)
    # perform forward prop
    outputs = mlp.forward_propagate(inputs)
    # print results
    print(f"The network input is: {inputs}")
    print(f"The network output is: {outputs}")