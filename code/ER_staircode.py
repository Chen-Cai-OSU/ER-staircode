import copy
import pickle
import sys
import pprint

import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from code.gen.ptc_model import toy_dataset, woojin2
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

BACKEND = 'multiprocessing'
linkage_kwargs = {'distance_threshold': 0, 'n_clusters': None, 'linkage': 'single'}

def density(data, bw=0.5):
    """
    Gaussian kernel density estimation
    :param data: np.array of shape (n, d)
    :return: density for each point
    """
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
    return kde.score_samples(data)

def slice_dgm_(model, pts, f, viz = False, **kwargs):
    """
    :param model: AgglomerativeClustering model
    :param pts: np.array of shape (n, d)
    :param f: np.array of shape (n, )
    :param viz: visualize dendrogram
    :param kwargs: kwargs for viz
    :return: a slice of ER-staircode that will be assembled later
    """
    counts = np.zeros(model.children_.shape[0])
    n_pts = len(model.labels_)
    decoration = {}

    for i in range(n_pts):
        decoration[i] = {'type': 'leaf', 'not_conquered': i}

    for i, merge in enumerate(model.children_):
        i_ = i + n_pts
        assert i_ not in decoration.keys(), f'key {i} already exists'
        dict_i = {}
        c1, c2 = list(merge)
        dict_i['type'] = 'non_leaf'
        dict_i['children'] = [c1, c2]
        dict_i['idx'] = i_

        tmp1, tmp2 = decoration[c1]['not_conquered'], decoration[c2]['not_conquered']
        tmp_max = tmp1 if f[tmp1] > f[tmp2] else tmp2
        tmp_min = tmp1 if f[tmp1] < f[tmp2] else tmp2
        dict_i['conquered'] = tmp_max # max(tmp1, tmp2)
        dict_i['conquered_pt'] = tuple(pts[tmp_max, :])
        dict_i['not_conquered'] = tmp_min # min(tmp1, tmp2)
        dict_i['height'] = model.distances_[i]
        decoration[i_] = dict_i

    # Plot the corresponding dendrogram
    if viz:
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)
    return decoration

def slice_dgm(data, f):
    """ a wrapper """
    model = AgglomerativeClustering(**linkage_kwargs)
    model = model.fit(data)
    decoration = slice_dgm_(model, data, f)
    return decoration

def assemble(stairs, f):
    """
    :param stairs: a list of dict of form
  0: {'not_conquered': 0, 'type': 'leaf'},
  1: {'not_conquered': 1, 'type': 'leaf'},
  2: {'not_conquered': 2, 'type': 'leaf'},
  3: {'children': [0, 2],
      'conquered': 2,
      'conquered_pt': (0.0, 1.5),
      'height': 1.5,
      'idx': 3,
      'not_conquered': 0,
      'type': 'non_leaf'},
  4: {'children': [1, 3],
      'conquered': 1,
      'conquered_pt': (2.0, 1.5),
      'height': 2.0,
      'idx': 4,
      'not_conquered': 0,
      'type': 'non_leaf'}}
    :param f: a sorted array of shape (n, 1)
    :return: a dict of form
    {(0.0, 1.5): {2: 1.5, 3: 1.5, 4: 1.0, 5: 0.5},
     (0.5, 1.5): {3: 0.5, 4: 0.5, 5: 0.5},
     (1.0, 1.5): {4: 0.5, 5: 0.5},
     (1.5, 1.5): {5: 0.5},
     (2.0, 1.5): {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.5, 5: 1.5}}
    """

    assert len(stairs) == f.shape[0]-1, f'len of stairs is {len(stairs)}. shape of f is {f.shape}'
    f = f[1:].tolist()
    stairs = dict(zip(f, stairs))
    I_x = {}

    for k, v in stairs.items():
        sigma = k
        for v_ in v.values():
            if v_['type'] == 'leaf':
                continue
            else: # non-leaf case
                idx = v_['conquered_pt']
                if idx not in I_x.keys(): I_x[idx] = {}
                I_x[idx][sigma] = v_['height']

    if args.verbose: pprint.pprint(I_x)
    return I_x

def plot_Ix(I_x):
    # I1 = I_x[1]
    random_key = list(I_x.keys())[0]
    I1 = I_x[random_key]
    pts = []
    for k, v in I1.items():
        pts.append([k, v])
    pts = np.array(pts)
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.title(f'I_{random_key}')
    plt.show()

def export(I_x):
    pickle.dump(I_x, open("I_x.pkl", "wb"))
    # return I_x_reload


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--n_jobs", default=-1, type=int, help='num of points')
parser.add_argument("--name", default='blob', type=str, help='name of dataset')
parser.add_argument("--bw", default=0.1, type=float, help='bandwidth for Gaussian kernel density estimation')

parser.add_argument("--d", default=2, type=int, help='dimension')
parser.add_argument("--full", action='store_true', help='repeat for all slices')
parser.add_argument("--verbose", action='store_true', help='for debug')
parser.add_argument("--parallel", action='store_true', help='repeat for all slices in parallel')
parser.add_argument("--woojin", action='store_true', help='Woojin\'s Toy example for verification')
parser.add_argument("--bs", default=-1, type=int, help='batch size')

if __name__ == '__main__':
    args = parser.parse_args()

    X_origin, _, distm = toy_dataset(n_sample=args.n, name=args.name)  # [[-3, 3], [0, 0], [-4.5, 1.5], [1, 1], [0, -4]]
    print(f' ave dist for is {np.average(distm)}')
    if args.woojin:
        X_origin, _, _, f = woojin2(switch=False)

    n_pt = X_origin.shape[0]
    if not args.woojin: f = - density(X_origin, bw=args.bw).reshape(n_pt, )
    f = f.reshape(n_pt, )
    if args.verbose: print(f'X_origin + f = \n {np.concatenate((X_origin, f.reshape((n_pt, 1))), axis=1)}')

    f_inds = f.argsort()
    X = X_origin[f_inds]
    f = f[f_inds]
    if args.verbose: print(f'After sorting: X + f = \n {np.concatenate((X, f.reshape((n_pt, 1))), axis=1)}')
    # print(f'shape of x and f is {X.shape}/{f.shape}')
    # print(f'After sorting: X + f = \n {np.concatenate((X, f.reshape((n_pt, 1))), axis=1)}')

    if args.verbose:
        print(f'X is {X} \n f_sort is {f}')
        print('-'*100)

    if args.parallel:
        bs =  'auto' if args.bs == -1 else args.bs
        datalist = [X[:i, :] for i in range(2, n_pt)]
        stairs = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(slice_dgm)(data) for data in datalist)
        pprint.pprint((stairs))
        sys.exit()

    if args.full:
        stairs = []
        for i in range(2, n_pt+1):
            X_ = X[:i, :]
            model = AgglomerativeClustering(**linkage_kwargs)
            model = model.fit(X_)
            decoration = slice_dgm_(model, X_, f[:i], truncate_mode='level', p=30)
            stairs.append(decoration)
        if args.verbose: pprint.pprint(stairs)
        I_x = assemble(stairs, f)
        export(I_x)
        # assert I_x == I_x_reload
        if args.verbose:
            pprint.pprint(I_x)
            plot_Ix(I_x)

    else:
        model = AgglomerativeClustering(**linkage_kwargs)
        model = model.fit(X)
        decoration = slice_dgm_(model, X, f, truncate_mode='level', p=30)
        pprint.pprint((decoration))
    sys.exit()
