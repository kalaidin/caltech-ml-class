"""
Linear Regression.
CS156 homework #2.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from input_data import TrainingSet, classify


class LR:
    def __init__(self):
        pass

    def run(self, X, Y):
        """Compute pseudo-inverse of X then multiply to Y."""
        self.weights = np.dot(np.linalg.pinv(X), Y)

    def predict(self, x):
        """Predict output on x."""
        return classify(np.dot(np.transpose(self.weights), x))

    def get_ein(self, X, Y):
        """Compute Ein on X. Question 5."""
        e_in = 0.0
        for i in range(X.shape[0]):
            if self.predict(X[i, :]) != Y[i]:
                e_in += 1
        return e_in / X.shape[0]


if __name__ == '__main__':
    training_set = TrainingSet(100)
    training_set.generate_data()
    training_set.split_set()

    lr = LR()
    lr.run(training_set.X, training_set.Y)
    print lr.get_ein(training_set.X, training_set.Y)

    training_set.plot()
    plt.show()
