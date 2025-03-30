import numpy as np


class SimpleNeuralNetwork:
    """
    Simple neural network that checks if a given binary representation of a positive number is even
    """

    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Sigmmoid function - smooth function that maps any number to a number from 0 to 1
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        """
        Derivative of sigmoid function
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, train_input, train_output, train_iters):
        for _ in range(train_iters):
            propagation_result = self.propagation(train_input)
            self.backward_propagation(
                propagation_result, train_input, train_output)

    def propagation(self, inputs):
        """
        Propagation process
        """
        return self.sigmoid(np.dot(inputs.astype(float), self.weights))

    def backward_propagation(self, propagation_result, train_input, train_output):
        """
        Backward propagation process
        """
        error = train_output - propagation_result
        self.weights += np.dot(
            train_input.T, error * self.d_sigmoid(propagation_result)
        )




if __name__ == '__main__':
    network = SimpleNeuralNetwork()
    print(network.weights)
    train_inputs = np.array(
        [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], ]
    )
    train_outputs = np.array([[1, 0, 1, 1, 0, 1]]).T
    train_iterations = 50000
    network.train(train_inputs, train_outputs, train_iterations)
    print(network.weights)
    print("Testing the data")
    test_data = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], ])
    for data in test_data:
        print(f"Result for {data} is:")
        print(network.propagation(data))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
