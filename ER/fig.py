import pprint
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

try:
    from ER.gen.ptc_model import toy_dataset, viz_pd
except ModuleNotFoundError:
    from gen.ptc_model import toy_dataset, viz_pd


def new_density(data, bw=0.5):
    distm = squareform(pdist(data, 'euclidean'))
    d = np.sum(np.exp(-np.multiply(distm, distm) / (2 * bw)), axis=0)
    return d


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
      'conquered_pt': (0.0, 1.5),   # the coordinates of conquered point
      'height': 1.5,                # the height of the current point
      'idx': 3,                     # the index of the current point
      'not_conquered': 0,           # the pt id in the clustering tree that hasn't been conquered yet
      'type': 'non_leaf'},          # whether the point stands for a leavf or not in the clustering tree

    """

    n_pts = len(model.labels_)
    decoration = {}

    for i in range(n_pts):
        decoration[i] = {'type': 'leaf', 'not_conquered': i}

    for i, merge in enumerate(model.children_):
        i_ = i + n_pts  # merge points
        # assert i_ not in decoration.keys(), f'key {i} already exists'
        dict_i = {}
        c1, c2 = list(merge)  # children 1 and children 2
        dict_i['type'] = 'non_leaf'
        tmp1, tmp2 = decoration[c1]['not_conquered'], decoration[c2]['not_conquered']
        (tmp_min, tmp_max) = (tmp1, tmp2) if f[tmp1] < f[tmp2] else (
            tmp2, tmp1)  # tmp_min/tmp_max according to function value

        # dict_i['conquered'] = tmp_max
        dict_i['conquered_pt'] = tuple(pts[tmp_max, :])  # kill/conquer the point with larger value
        dict_i['not_conquered'] = tmp_min  # the point with smaller value is not conquered yet
        dict_i['height'] = model.distances_[
            i]  # important: i instead of i_. model distances_ is an array of shape (n_child,)

        decoration[i_] = dict_i

    return decoration


def assemble(stairs, f, verbose=False):
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
            KEY is the coordinates of each point
            VALUE is a dict (I_xi) where key is sigma(function value) and value is epsilon (function value)
    {(0.0, 1.5): {2.0: 1.5, 3.0: 1.5, 4.0: 1.0, 5.0: 0.5},
     (0.5, 1.5): {3.0: 0.5, 4.0: 0.5, 5.0: 0.5},
     (1.0, 1.5): {4.0: 0.5, 5.0: 0.5},
     (1.5, 1.5): {5.0: 0.5},
     (2.0, 1.5): {1.0: 2.5, 2.0: 2.0, 3.0: 1.5, 4.0: 1.5, 5.0: 1.5}}
    """

    assert len(stairs) == f.shape[0] - 1, f'len of stairs is {len(stairs)}. shape of f is {f.shape}'
    f = f[1:].tolist()
    stairs = dict(zip(f, stairs))
    I_x = {}

    for sigma, v in stairs.items():
        for v_ in v.values():
            if v_['type'] == 'leaf':
                continue
            else:  # non-leaf case
                idx = str(v_['conquered_pt'])  # convert tuple to string for json.dump. idx is of form (0.0, 1.5).
                if idx not in I_x: I_x[idx] = {}
                I_x[idx][sigma] = v_['height']

    if verbose: pprint.pprint(I_x)
    return I_x


def plot_first_block(f, color, ylimit=10, xlimit=10, cmap=None):
    pts = [(min(f), ylimit), (xlimit, 0)]
    pts = np.array(pts)
    c = color[np.argmax(f)] if cmap is None else cmap(0)
    mappable = plt.fill_between(pts[:, 0], pts[:, 1], step="post", alpha=0.5, color=c, edgecolor='blue')


def plot_Ix(I_x, key=None, ext=1, show=False, pointsize=0, title=None,
            c='blue', fname=None, xlimit=None, cmap=None, ylimit=None):
    """
    plot a single staircase
    :param I_x:
    :param key:
    :param ext: extension to the right
    :param cmap: if none, use discrete color for different clusters; otherwise, use cont' color map.
    :return:
    """

    if key is None: key = list(I_x.keys())[0]
    I_key = I_x[key]

    pts = []
    for sig, ep in I_key.items():
        pts.append([sig, ep])
    if xlimit is None:
        pts.append([sig + ext, ep])  # add a bit extension
    else:
        assert xlimit >= sig + ext, f'xlimit ({xlimit}) < sig+ext ({sig + ext})'
        pts.append([xlimit, ep])

    pts = np.array(pts)
    if pointsize: plt.scatter(pts[:, 0], pts[:, 1], s=pointsize)
    if cmap is None:
        mappable = plt.fill_between(pts[:, 0], pts[:, 1], step="post", alpha=0.3, color=c, edgecolor='blue')
    else:
        # https://bit.ly/3o5rgfB
        ymax = max(I_key.values())
        c = np.array([ymax] * len(pts))
        mappable = plt.fill_between(pts[:, 0], pts[:, 1], step="post", alpha=0.3, color=cmap(c), edgecolor='blue')
    if title: plt.title(title)

    if show: plt.show()
    if fname is not None: plt.savefig(fname, bbox_inches='tight')
    return mappable


parser = ArgumentParser("ER", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--name", default='blob', type=str, help='name of dataset')
parser.add_argument("--bw", default=0.1, type=float, help='bandwidth for Gaussian kernel density estimation')
parser.add_argument("--seed", default=42, type=int, help='random seed to generate data')
parser.add_argument("--verbose", action='store_true', help='for debug')
parser.add_argument("--xlimit", default=5.5, type=float, help='xlimit')
parser.add_argument("--ylimit", default=13, type=float, help='ylimit')

if __name__ == '__main__':
    args = parser.parse_args()

    # generate 2d point clouds and codensity
    X_origin, color, _ = toy_dataset(n_sample=args.n, name=args.name, seed=args.seed)
    n_pt = X_origin.shape[0]
    f_origin = - new_density(X_origin, bw=args.bw).reshape(n_pt, )
    f_inds = f_origin.argsort()
    X, f = X_origin[f_inds], f_origin[f_inds]

    # compute ER staircode
    linkage_kwargs = {'distance_threshold': 0, 'n_clusters': None, 'linkage': 'single'}
    stairs = []
    for i in range(2, n_pt + 1):
        X_ = X[:i, :]
        model = AgglomerativeClustering(**linkage_kwargs)
        model = model.fit(X_)
        decoration = slice_dgm_(model, X_, f[:i])
        stairs.append(decoration)
    I_x = assemble(stairs, f)

    # visulization
    cmap = sns.cubehelix_palette(as_cmap=True)
    viz_pd(X_origin, color=f_origin, show=True, colorbar=True)
    fig, ax = plt.subplots(figsize=(5, 6))
    plot_first_block(f, color, xlimit=args.xlimit, ylimit=args.ylimit, cmap=cmap)
    for i, (pt, c) in enumerate(zip(X_origin.tolist(), color)):
        try:
            mappable = plot_Ix(I_x, key=str(tuple(pt)), c=c, xlimit=args.xlimit, ext=0, cmap=cmap)
        except KeyError:
            print(f'{i}-th point {pt} not in I_x')
    plt.show()
