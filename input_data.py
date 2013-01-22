"""
Random data generator.
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def classify(x):
    return 1 if x > 0 else -1


class TrainingSet:
    def __init__(self, N=10):
        self.N = N
        self.X = np.empty([N, 3])
        self.Y = np.empty([N])

    def generate_data(self):
        """Generate training set."""
        for i in range(self.N):
            self.X[i][0] = 1
            self.X[i][1] = random.uniform(-1, 1)
            self.X[i][2] = random.uniform(-1, 1)

    def split_set(self):
        """Label the set using a generated random line."""
        x1 = -1
        y1 = random.uniform(-1, 1)
        x2 = 1
        y2 = random.uniform(-1, 1)
        self.f_points = [[x1, x2], [y1, y2]]
        f = [x2 - x1, y2 - y1]
        f_norm = [y1 - y2, x2 - x1]
        for i in range(self.N):
            vector_to_dot = [self.X[i][1] - x1, self.X[i][2] - y1]
            self.Y[i] = classify(np.dot(f_norm, vector_to_dot))

    def plot(self):
        """Plot training data."""
        for i in range(self.N):
            plt.plot(self.X[i, 1],
                     self.X[i, 2], 'rx' if self.Y[i] == 1 else 'bx')
        # Show a line
        #plt.plot(self.f_points[0], self.f_points[1])
