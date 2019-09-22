import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors.kde import KernelDensity
import sys
import networkx as nx
import scipy
from Esme.helper.time import timefunction
import seaborn as sns


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

def viz_pd(pts, show=False, color=None):
    """
    :param pts: np.array of shape (n, 2)
    :return:
    """
    cmap = sns.cubehelix_palette(as_cmap=True)
    if color is None: color = np.random.rand(1, pts.shape[0])
    color = color.reshape((pts.shape[0],))

    f, ax = plt.subplots()
    points = ax.scatter(x=pts[:, 0], y=pts[:, 1], s=5, c = list(color))
    f.colorbar(points)
    if show: plt.show()

def ambient_noise(x_range = (0, 1), y_range = (0,1), n = 100):
    coords = [[random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])] for _ in range(n)]
    return np.array(coords)

def density(data):
    """
    :param data: np.array of shape (n, d)
    :return: density for each point
    """
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    return kde.score_samples(data)

def ex():
    # https://stackoverflow.com/questions/39735147/how-to-color-matplotlib-scatterplot-using-a-continuous-value-seaborn-color
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    x, y, z = np.random.rand(3, 100)
    cmap = sns.cubehelix_palette(as_cmap=True)

    f, ax = plt.subplots()
    points = ax.scatter(x, y, c=z, s=50)
    f.colorbar(points)
    plt.show()

@timefunction
def p(points, f, x1, x2):
    """
    :param points: array of shape (n, d)
    :param f: array of shape (n, 1)
    :param x1: idx
    :param x2: idx
    :return: (sigma, epsilon)
    """
    assert f.shape[1] == 1

    sigma = max(f[x1][0], f[x2][0])
    # points = np.random.random((100,3))
    distm = scipy.spatial.distance.pdist(points)
    distm = scipy.spatial.distance.squareform(distm)
    n = points.shape[0]

    filter_m = np.outer([f < sigma + 1e-6], [f < sigma + 1e-6]) # todo: viz filter_m should be square
    val = np.multiply(distm, filter_m)
    edgelist = np.argwhere(val > 0) # np.array of shape (_, 2)
    val = val[val> 0]

    n_ = edgelist.shape[0]
    assert edgelist.shape[0] == val.shape[0]
    edgelist = np.concatenate((edgelist, val.reshape((n_,1))), axis=1) # array of shape (_, 3)
    assert edgelist.shape[1] == 3

    # format data for nx
    lines = []
    for i in range(n_):
        s, t, weight = edgelist[i][0], edgelist[i][1], edgelist[i][2]
        line = f"{int(s)} {int(t)}" + ' {' + f"'weight':{float(weight)}" + '}'
        lines.append(line)

    G = nx.parse_edgelist(lines, nodetype=int)
    # list(G)
    # list(G.edges(data=True))
    mst = nx.minimum_spanning_tree(G)
    # x1, x2 = 1, 10
    path = nx.shortest_path(mst, x1, x2, weight='weight') # [1, 70, 78, 13, 10]
    path_length_lis = [mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
    epsion = 0 if len(path_length_lis) == 0 else max(path_length_lis)

    return sigma, epsion

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=1, type=int, help='point index') # (1515, 500) (3026,)
parser.add_argument("--n", default=30, type=int, help='point index') # (1515, 500) (3026,)


if __name__ == '__main__':
    # ex()
    # sys.exit()
    args = parser.parse_args()

    points_1 = pd_from_cycle(n = args.n, center=(0,0))
    points_2 = pd_from_cycle(n = args.n, center=(5, 0))
    noise = ambient_noise(x_range=(-1,6), y_range=(-1,1))

    points = np.concatenate((points_1, points_2, noise))
    print(points.shape)
    density_score_ = density(points).reshape(len(points),1)
    density_score = density(points).reshape(len(points),)

    stair_case = []
    for i in range(args.n):
        sigma, epsilon = p(points, density_score_, args.idx, i)
        stair_case.append([sigma, epsilon])
    stair_case = np.array(stair_case)
    print(stair_case)

    f, ax = plt.subplots()
    ax.scatter(x = stair_case[:,0], y = stair_case[:,1])
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_title(f'Staircase for index {args.idx}')
    plt.show()

    print(density_score.shape)
    viz_pd(points, show=True, color=density_score)
