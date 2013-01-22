"""
Perceptron.
CS156 homework #1.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from input_data import TrainingSet, classify


class Perceptron:
    def __init__(self):
        self.weights = np.zeros([3])

    def response(self, x):
        """
        Compute perceptron response.
        Note the bias component is added (1).
        """
        return classify(np.dot(self.weights, x))

    def update_weights(self, x, error):
        """Update weights."""
        self.weights += error * x

    def learn(self, X, Y):
        """Run PLA."""
        done = False
        all_classified = False
        iteration = 0
        while not all_classified:
            all_classified = True
            for i in range(X.shape[0]):
                if Y[i] != self.response(X[i, :]):
                    error = Y[i] - self.response(X[i, :])
                    self.update_weights(X[i, :], error)
                    all_classified = False
            # Plot intermediate results
            #self.plot()
            iteration += 1
        print("Done in %i iterations." % (iteration))

    def plot(self):
        """
        Plot the result.
        The resulting equation is:
        self.weights[1] * x + self.weights[2] * y + self.weights[0] = 0,
        which means:
        y = (- self.weights[0] - self.weights[1] * x) / self.weights[2].
        Compute this function in x = -1 and x = 1 and plot it.
        """
        plt.plot([-1, 1],
                 [(-self.weights[0] - self.weights[1] * -1) / self.weights[2],
                 (-self.weights[0] - self.weights[1] * 1) / self.weights[2]],
                 '--k')


if __name__ == '__main__':
    training_set = TrainingSet(100)
    training_set.generate_data()
    training_set.split_set()

    perceptron = Perceptron()
    perceptron.learn(training_set.X, training_set.Y)

    training_set.plot()
    perceptron.plot()
    plt.show()
