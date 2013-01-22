"""
Homework assignment #2, questions 1-2.
"""

import random
import numpy as np


def flip():
    return 1 if random.uniform(0, 1) > 0.5 else 0


def simulate(coin_number=1000, flips=10, iterations=10000):
    v = np.empty(3)
    for n in range(iterations):
        coins = np.zeros(coin_number)
        for i in range(coin_number):
            for j in range(flips):
                coins[i] += flip()
            coins[i] = coins[i] / flips
        v[0] += coins[0]
        v[1] += coins[random.randint(0, coin_number - 1)]
        v[2] += min(coins)
    return np.multiply(v, 1.0 / iterations)


if __name__ == '__main__':
    print simulate()  # iteration=100000 takes too much time :-)
