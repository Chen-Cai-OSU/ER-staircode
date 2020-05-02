import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets


def viz_pd(pts, show=False, color=None, annotate=None, inter=False):
    """
    :param pts: np.array of shape (n, 2)
    :param show:
    :param color:
    :param annotate: point idx for annotation
    :return:
    """
    if annotate is list:
        for idx in annotate: assert idx <= pts.shape[0]

    cmap = sns.cubehelix_palette(as_cmap=True)
    if color is None: color = np.random.rand(1, pts.shape[0])
    color = color.reshape((pts.shape[0],))

    f, ax = plt.subplots()
    points = ax.scatter(x=pts[:, 0], y=pts[:, 1], s=5, c=list(color))
    f.colorbar(points)

    if annotate is not None:
        for idx in annotate:
            ax.text(pts[idx, 0] + 0.1, pts[idx, 1] + 0.1, idx, fontsize=9)

    if show: plt.show()


def pd_from_cycle(n=100, center=(0, 0)):
    np.random.seed(42)
    random.seed(42)
    pds = []
    for i in range(n):
        length = np.sqrt(np.random.uniform(0.9, 1))
        angle = np.pi * np.random.uniform(0, 2)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        pds.append([x + center[0], y + center[1]])
    pd = np.array(pds)
    return pd


def pts2pd(pts, color):
    # used for interactive plot
    assert pts.shape[1] == 2
    x, y = pts[:, 0], pts[:, 1]
    pd_frame = pd.DataFrame(data={'label': color, 'x': x, 'y': y, 'idx': range(pts.shape[0])})
    return pd_frame


def ambient_noise(x_range=(0, 1), y_range=(0, 1), n=100):
    coords = [[random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])] for _ in range(n)]
    return np.array(coords)


def two_cycle(n=20):
    """ generate two cycles of size n and noise of n """
    points_1 = pd_from_cycle(n=n, center=(0, 0))
    points_2 = pd_from_cycle(n=n, center=(5, 0))
    noise = ambient_noise(x_range=(-1, 6), y_range=(-1, 1), n=n)
    points = np.concatenate((points_1, points_2, noise))  # array of shape (n, d)

    distm = pdist(points)
    distm = squareform(distm)

    return points, distm


def uniform2d(n=100):
    points = np.random.random((n, 2))
    distm = pdist(points)
    distm = squareform(distm)
    return points, distm


def point_online(uniform=True):
    if uniform:
        xrange = np.linspace(0, 1, 11)
    else:
        xrange = np.array(np.linspace(0, 0.5, 7).tolist() + np.linspace(0.6, 1, 4).tolist())

    points = []
    for x in xrange:
        points.append([x, 0.5])

    points = np.array(points)
    distm = pdist(points)
    distm = squareform(distm)
    return points, distm


def pts2distm(pts, metric='euclidean'):
    if metric == 'random':
        np.random.seed(42)
        n = pts.shape[0]
        distm = np.random.random((n, n))
        distm = (distm + distm.T) * 0.5
        distm -= np.diag(distm.diagonal())
        return distm

    distm = pdist(pts, metric=metric)
    distm = squareform(distm)
    return distm


def modelnet(idx=100):
    """
    :param idx:
    :return: idx-th modelnet 10 pointclouds
    """
    from Esme.graph.dataset.modelnet import load_modelnet
    train_dataset, test_dataset = load_modelnet('10', point_flag=True)
    all_dataset = train_dataset + test_dataset
    labels = [int(data.y.numpy()) for data in all_dataset]

    data = all_dataset[idx]
    data = data.pos.numpy()
    print(data.shape)

    distm = pdist(data)
    distm = squareform(distm)

    return data, labels[idx], distm


def woojin():
    x1 = (0, 0)
    x2 = (0, 3)
    x3 = (4, 3)
    pts = [x1, x2, x3]
    for i in range(1, 8):
        pts.append((0.5 * i, 3))

    data = np.array(pts)
    f = np.array([0, 1, 1.1] + [2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07]).reshape(len(pts), 1)
    color = ['b', 'r', 'y'] + ['grey'] * 7

    distm = pdist(data)
    distm = squareform(distm)

    return data, color, distm, f


def woojin2(switch=False):
    x1 = (0, 0)
    x2 = (0, 1.5)
    x3 = (2, 1.5)
    pts = [x1, x2, x3]
    for i in range(1, 4):
        pts.append((0.5 * i, 1.5))

    data = np.array(pts)
    f = np.array([0, 2, 1] + [3, 4, 5]).reshape(len(pts), 1)
    if switch:
        f = np.array([0, 1, 2] + [3, 4, 5]).reshape(len(pts), 1)
    color = ['b', 'r', 'y'] + ['grey'] * 3

    distm = pdist(data)
    distm = squareform(distm)

    return data, color, distm, f


def woojin3():
    # example for check with I_x with revet
    x1 = (0, 0)
    x2 = (4, 3)
    x3 = (0, 3)
    x4 = (2.5, 3)
    pts = [x1, x2, x3, x4]

    data = np.array(pts)
    f = np.array([0, 1, 2, 3]).reshape(len(pts), 1)
    color = ['b'] * 4

    distm = pdist(data)
    distm = squareform(distm)

    return data, color, distm, f


def pts_on_square(add_noise=False, n_noise=5):
    pts = []
    for theta in np.linspace(0, 2 * np.pi, 200):
        pts.append((np.cos(theta), np.sin(theta)))
    noise = ambient_noise(x_range=(-1, 1), y_range=(-1, 1), n=n_noise)
    if add_noise: pts = np.concatenate((pts, noise))

    distm = pdist(pts)
    distm = squareform(distm)

    return np.array(pts), distm


def color_map(n):
    if n == 0: return 'b'
    if n == 1: return 'r'
    if n == -1: return 'y'
    if n == 2: return 'g'


def toy_dataset(n_sample=200, name='blob', seed=42, pd=False, metric='euclidean', scale=1, **kwargs):
    """

    :param n_sample:
    :param name:
    :param seed:
    :param pd:
    :param metric:
    :param scale:
    :param kwargs:
    :return: np.array of shape (n, d),  color, pairwise distance matrix
    """
    if name == 'circle':
        pts, color = datasets.make_circles(n_samples=n_sample, factor=.5, noise=.05,
                                           random_state=seed)  # both pts and color are array
    elif name == 'moon':
        pts, color = datasets.make_moons(n_samples=n_sample, noise=.05, random_state=seed)
    elif name == 'blob':
        pts, color = datasets.make_blobs(n_samples=n_sample, random_state=seed, n_features=kwargs.get('n_features', 2))
    elif name == 'aniso':
        X, color = datasets.make_blobs(n_samples=n_sample, random_state=seed)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        pts = np.dot(X, transformation)
    elif name == 'digits':
        digits = datasets.load_digits()
        pts, color = digits.images, digits.target
        pts = pts.reshape(1797, 64)
        sample_idx = np.random.randint(0, 1797, 200)
        pts, color = pts[sample_idx, :], color[sample_idx]
    elif name == 'test':
        pts = [[-3, 3], [0, 0], [-4.5, 1.5], [1, 1], [0, -4]]
        return np.array(pts), None, None


    else:
        raise Exception(f'No dataset {name}')

    if pd:
        return pts2pd(pts, color)

    color = [color_map(c) for c in color]
    distm = pts2distm(pts, metric=metric)
    pts, distm = pts * scale, distm * scale

    return np.array(pts), color, distm


def uniform_noise(xrange=(0, 1), yrange=(0, 1), size=100):
    """
    uniform 2d noise
    :param xrange:
    :param yrange:
    :param size: num of noisy points
    :return:
    """
    x = np.random.uniform(low=xrange[0], high=xrange[1], size=(size, 1))
    y = np.random.uniform(low=yrange[0], high=yrange[1], size=(size, 1))
    pts = np.concatenate((x, y), axis=1)
    color = ['grey'] * size
    return pts, color


def non_uniform(n=100, noise=False, d=2):
    mean = np.array(range(1, d + 1))
    cov = np.identity(d) * 0.01
    pts1 = np.random.multivariate_normal(mean, cov, n)

    mean = -mean
    cov = np.identity(d) * 0.1
    pts2 = np.random.multivariate_normal(mean, cov, n)

    noise_pts = []
    if noise:
        for i in np.linspace(-1.1, 1.1, 30):
            noise_pts.append((i, 2 * i))
        noise_pts = np.array(noise_pts)
        pts = np.concatenate((pts1, pts2, noise_pts), axis=0)
        color = ['red'] * n + ['blue'] * n + ['grey'] * 30
    else:
        pts = np.concatenate((pts1, pts2), axis=0)
        color = ['red'] * n + ['blue'] * n

    distm = pdist(pts)
    distm = squareform(distm)

    return np.array(pts), color, distm


if __name__ == '__main__':
    pts, _ = uniform_noise(size=1000)
    viz_pd(pts, show=True)
    sys.exit()

    pts = pts_on_square(add_noise=True)
    viz_pd(pts, show=True)

    sys.exit()
    points, label = modelnet()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    sys.exit()

    points, distm = uniform(5000)
    viz_pd(points, show=True)

    print(np.mean(points), np.mean(distm))
