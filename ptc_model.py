import random
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
np.random.seed(42)
random.seed(42)

def pd_from_cycle(n = 100, center = (0, 0)):
    pds = []
    for i in range(n):
        length = np.sqrt(np.random.uniform(0.9, 1))
        angle = np.pi * np.random.uniform(0, 2)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        pds.append([x + center[0], y + center[1]])
    pd = np.array(pds)
    return pd

def ambient_noise(x_range = (0, 1), y_range = (0,1), n = 100):
    coords = [[random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])] for _ in range(n)]
    return np.array(coords)

def two_cycle(n = 20):
    """ generate two cycles of size n and noise of n """
    points_1 = pd_from_cycle(n=n, center=(0, 0))
    points_2 = pd_from_cycle(n=n, center=(5, 0))
    noise = ambient_noise(x_range=(-1, 6), y_range=(-1, 1), n=n)
    points = np.concatenate((points_1, points_2, noise))  # array of shape (n, d)

    distm = pdist(points)
    distm = squareform(distm)

    return points, distm

if __name__ == '__main__':
    points, distm = two_cycle(50)
    print(np.mean(points), np.mean(distm))
