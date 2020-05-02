import pickle
import pprint
import sys
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KernelDensity

from ER.gen.ptc_model import toy_dataset
from ER.helper.format import precision_format as pf

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


def slice_dgm_(model, pts, f, viz=False, **kwargs):
    """
    :param model: AgglomerativeClustering model
    :param pts: np.array of shape (n, d)
    :param f: np.array of shape (n, )
    :param viz: visualize dendrogram
    :param kwargs: kwargs for viz
    :return: a vertical slice of ER-staircode that will be assembled later.
            a dict of the following form (KEY: VALUE), where KEY is point_id/merge_point ()
            and VALUE is of the following form

      {'children': [0, 2],          # indices of point that are children of the current point
      'conquered': 2,               # the index of point that is being conquered
      'conquered_pt': (0.0, 1.5),   # what is this? # todo: the coordinates of conquered point
      'height': 1.5,                # the height of the current point
      'idx': 3,                     # the index of the current point
      'not_conquered': 0,           # the pt id in the clustering tree that hasn't been conquered yet
      'type': 'non_leaf'},          # whether the point stands for a leavf or not in the clustering tree

    """

    t0 = time()
    n_pts = len(model.labels_)
    decoration = {}

    for i in range(n_pts):
        decoration[i] = {'type': 'leaf', 'not_conquered': i}

    for i, merge in enumerate(model.children_):
        i_ = i + n_pts # merge points
        # assert i_ not in decoration.keys(), f'key {i} already exists'
        dict_i = {}
        c1, c2 = list(merge) # children 1 and children 2
        dict_i['type'] = 'non_leaf'
        tmp1, tmp2 = decoration[c1]['not_conquered'], decoration[c2]['not_conquered']
        (tmp_min, tmp_max) = (tmp1, tmp2) if f[tmp1] < f[tmp2] else (tmp2, tmp1) # tmp_min/tmp_max according to function value

        # dict_i['conquered'] = tmp_max
        dict_i['conquered_pt'] = tuple(pts[tmp_max, :]) # kill/conquer the point with larger value
        dict_i['not_conquered'] = tmp_min               # the point with smaller value is not conquered yet
        dict_i['height'] = model.distances_[i]          #

        decoration[i_] = dict_i

    # print(f'slice_dgm_ takes {time()-t0} for {len(model.children_)}')
    return decoration


def slice_dgm(data, f):
    """ a wrapper """
    model = AgglomerativeClustering(**linkage_kwargs)
    model = model.fit(data)
    decoration = slice_dgm_(model, data, f)
    return decoration


def slice_dgm2(i):
    """ a wrapper of slice_dgm_ """
    assert 'X' in globals().keys()
    assert 'f' in globals().keys()
    global f
    global X
    data_local = deepcopy(X[:i, :])
    f_local = deepcopy(f[:i])
    model = AgglomerativeClustering(**linkage_kwargs)
    model = model.fit(data_local)
    decoration = slice_dgm_(model, data_local, f_local)
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

    :return: I_x: a dict of form
    {(0.0, 1.5): {2: 1.5, 3: 1.5, 4: 1.0, 5: 0.5},
     (0.5, 1.5): {3: 0.5, 4: 0.5, 5: 0.5},
     (1.0, 1.5): {4: 0.5, 5: 0.5},
     (1.5, 1.5): {5: 0.5},
     (2.0, 1.5): {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.5, 5: 1.5}}
    """

    assert len(stairs) == f.shape[0] - 1, f'len of stairs is {len(stairs)}. shape of f is {f.shape}'
    f = f[1:].tolist()
    stairs = dict(zip(f, stairs))
    I_x = {}

    for k, v in stairs.items():
        sigma = k
        for v_ in v.values():
            if v_['type'] == 'leaf':
                continue
            else:  # non-leaf case
                idx = str(v_['conquered_pt'])  # convert tuple to string for json.dump. idx is of form (0.0, 1.5).
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
    """ export I_x """
    t0 = time()
    print('Finish Computing. Saving now...')
    with open('./I_x.pkl', 'wb') as f:
        pickle.dump(I_x, f)
    print(f'Finish saving. Takes {pf(time() - t0, 2)}')

    # import json
    # with open('./I_x.json', 'w') as f:
    #     json.dump(I_x, f)
    #
    # with open('./I_x.json', 'r') as f:
    #     I_x_reload = json.load(f)
    # pprint.pprint(I_x)
    # print()
    # pprint.pprint(I_x_reload)
    # assert I_x == I_x_reload


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--n_jobs", default=-1, type=int, help='num of jobs')
parser.add_argument("--name", default='moon', type=str, help='name of dataset')
parser.add_argument("--bw", default=0.1, type=float, help='bandwidth for Gaussian kernel density estimation')

parser.add_argument("--d", default=2, type=int, help='dimension')
parser.add_argument("--full", action='store_true', help='repeat for all slices')
parser.add_argument("--verbose", action='store_true', help='for debug')
parser.add_argument("--parallel", action='store_true', help='repeat for all slices in parallel')
parser.add_argument("--bs", default=-1, type=int, help='batch size')

if __name__ == '__main__':
    args = parser.parse_args()

    # generate 2d point clouds
    X_origin, _, _ = toy_dataset(n_sample=args.n, name=args.name)  # X_origin is of shape (n, 2)

    n_pt = X_origin.shape[0]
    f = - density(X_origin, bw=args.bw).reshape(n_pt, )
    if args.verbose: print(f'X_origin + f = \n {np.concatenate((X_origin, f.reshape((n_pt, 1))), axis=1)}')

    # sort by function value
    f_inds = f.argsort()
    X = X_origin[f_inds]
    f = f[f_inds]

    if args.verbose:
        print(f'After sorting: X + f = \n {np.concatenate((X, f.reshape((n_pt, 1))), axis=1)}')
        print('-' * 100)

    if args.parallel:
        bs = 'auto' if args.bs == -1 else args.bs
        # datalist = [X[:i, :] for i in range(2, n_pt)]
        stairs = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(slice_dgm2)(i) for i in range(2, n_pt + 1))
        # pprint.pprint((stairs))

    elif args.full:
        stairs = []
        for i in range(2, n_pt + 1):
            X_ = X[:i, :]
            model = AgglomerativeClustering(**linkage_kwargs)
            model = model.fit(X_)
            decoration = slice_dgm_(model, X_, f[:i])
            stairs.append(decoration)
    else:
        sys.exit()
        # single slice of ER-staircode
        model = AgglomerativeClustering(**linkage_kwargs)
        model = model.fit(X)
        decoration = slice_dgm_(model, X, f)
        pprint.pprint((decoration))

    if args.verbose:
        pprint.pprint(stairs)

    I_x = assemble(stairs, f)
    export(I_x)

    if args.verbose:
        pprint.pprint(I_x)
        plot_Ix(I_x)
