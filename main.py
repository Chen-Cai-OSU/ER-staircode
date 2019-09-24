import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors.kde import KernelDensity
import sys
import networkx as nx
import scipy
from Esme.helper.time import timefunction
import seaborn as sns
from ptc_model import two_cycle
from util import  get_left_epsilon, get_previous_epsilon
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

def density(data):
    """
    :param data: np.array of shape (n, d)
    :return: density for each point
    """
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    return kde.score_samples(data)


@timefunction
def _p(distm, f, x1, x2, sigma = None):
    """
    function used to compute p(x, x')
    :param points: array of shape (n, d)
    :param f: array of shape (n, 1)
    :param x1: idx
    :param x2: idx
    :return: (sigma, epsilon)
    """
    assert f.shape[1] == 1

    if sigma is None: sigma = max(f[x1][0], f[x2][0])

    # points under sigma
    tor = 1e-6
    filter_m = np.outer([f < sigma + tor], [f < sigma + tor]) # todo: viz filter_m should be square
    val = np.multiply(distm, filter_m)
    edgelist = np.argwhere(val > 0) # np.array of shape (_, 2)
    val = val[val> 0]

    n_ = edgelist.shape[0] # num of edges in sub-complete graph
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
    mst = nx.minimum_spanning_tree(G)
    path = nx.shortest_path(mst, x1, x2, weight='weight') # [1, 70, 78, 13, 10]
    path_length_lis = [mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
    epsion = 0 if len(path_length_lis) == 0 else max(path_length_lis)

    return sigma, epsion

def elbow_test(k):
    sigma_left, epsilon_left = _p(distm, cod_score_, args.idx, x_, sigma=cod_score_sort[k-1])
    sigma_right, epsilon_right = _p(distm, cod_score_, args.idx, x_, sigma=cod_score_sort[k + 1])
    if epsilon_right < epsilon_left:
        return True
    else:
        return False


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=6, type=int, help='point index') # (1515, 500) (3026,)
parser.add_argument("--n", default=50, type=int, help='point index') # (1515, 500) (3026,)
parser.add_argument("--method", default='heu', type=str, help='bf (brute force), heu (heuristic)') # (1515, 500) (3026,)



if __name__ == '__main__':
    # ex()
    # sys.exit()
    args = parser.parse_args()
    points, distm = two_cycle(n = args.n)

    cod_score_ = - density(points).reshape(len(points),1)
    cod_score_dict = {}
    for i in range(cod_score_.shape[0]):
        k, v = cod_score_[i][0], i
        assert k not in cod_score_dict.keys()
        cod_score_dict[k] = v
    assert len(cod_score_dict) ==  cod_score_.shape[0] # no duplicates

    cod_score_sort = [i[0] for i in cod_score_.tolist()]
    cod_score_sort.sort()
    density_score = density(points).reshape(len(points),)

    # compute p(x, x') where x = args.idx and x'=10
    x_ = 1

    try:
        assert cod_score_[x_][0] < cod_score_[args.idx][0]
    except AssertionError:
        sys.exit(f'x is {args.idx} with codensity {cod_score_[args.idx][0]}. x_ is {x_} with codensity {cod_score_[x_][0]}')

    start_idx, end_idx = cod_score_sort.index(cod_score_[args.idx][0]), len(cod_score_sort)
    print(f'start_idx is {start_idx}, end_idx is {end_idx}')
    sigma, epsilon = _p(distm, cod_score_, args.idx, x_, sigma=cod_score_sort[start_idx])
    stair = [(sigma, epsilon)]

    k = start_idx
    max_step_size = end_idx - k -1

    if args.method == 'heu':
        while max_step_size > 0:
            max_step_size = end_idx - k - 1
            for step_size in [max_step_size, max_step_size//2, max_step_size//4,4, 1]:
                sigma, epsilon = _p(distm, cod_score_, args.idx, x_, sigma=cod_score_sort[k+step_size])
                if epsilon == get_previous_epsilon(stair) or step_size==1:
                    print(f'previous idx is {k}. current idx is {k+step_size}')
                    k += step_size
                    stair.append((sigma, epsilon))
                    break

    elif args.method == 'bf':
        # brute force
        for k in range(start_idx, end_idx):
            sig = cod_score_sort[k]
            sigma, epsilon = _p(distm, cod_score_, args.idx, x_, sigma = sig)
            print(k, sigma, epsilon)
            stair.append([sigma, epsilon])

    stair.sort(key=lambda x: x[0])
    stair = np.array(stair)

    for i in range(stair.shape[0]):
        print(i, stair[i])

    f, ax = plt.subplots()
    ax.plot(stair[:,0], stair[:,1], 'b-')
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_title(f'p({args.idx}, {x_})')
    # plt.show()
    sys.exit()


    print(density_score.shape)
    viz_pd(points, show=True, color=density_score)
