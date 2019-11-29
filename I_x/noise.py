import copy
import os
import pickle
from pathlib import Path
from time import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from helper.format import precision_format as pf
from joblib import delayed, Parallel
from helper.viz import hist_plot

from I_x.I_x_slice import set_graph
from I_x.blank import msts
from I_x.dgm2d import get_idxsort, get_subgraph, update_subgraph, D_x_slice_clean, dgm_format, subslice, load_subgraph
from I_x.mst_test import get_ultra_matrix
from gen.example import density
from gen.ptc_model import toy_dataset, modelnet, woojin, non_uniform, uniform_noise, pts2distm, woojin2, woojin3
from gen.util import viz_stairs_, write_pts_for_viz
from helper.time import precision_format as pf
from viz import viz_pd
import json
import pickle
import os
import numpy as np
from helper.intersection import staircase

BACKEND = 'multiprocessing'

# todo: if noise is in test dir, there is some problem. If nosie.py is in I_x, everything is fine.

def D_x_slice(f, f_sort, idx_sort, sigma, sigmas, print_=False, mst_opt = False, step = 1, verbose = 1, **kwargs):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x_: idx
    :param sigma: sigma considered for now
    :param kwargs: kwargs for nx2gt conversion (get_ultra_matrix)
    :return: A slice of 2d diagram (a list of dict of form {x_: (sigma, epsilon)} )
    """

    t0 = time()
    n = f.shape[0]
    assert len(set(sigmas)) == len(sigmas) # assert no duplicates
    idx = sigmas.index(sigma)

    if mst_opt == False:
        G = load_subgraph(sigma, sigmas, print_=False)
        t1 = time()
        if print_: print(f'get_subgraph takes {pf(t1-t0,2)}')
        mst = nx.minimum_spanning_tree(G, weight='weight')
        # mst_total_lengh(mst)
        # mst_total_lengh(msts_list[idx])
    else:
        assert 'msts_list' in globals().keys()
        t1 = time()
        mst = msts_list[idx]

    t2 = time()
    if print_:
        if mst_opt == False: print(f'load_subgraph takes {pf(t1 - t0, 2)})')
        print(f'mst takes {pf(t2 - t1, 2)})')

    kwargs['sigma'] = sigma
    d, old2new_dict, new2old_dict = get_ultra_matrix(mst, n = n, fast=True, faster=True, step=100, **kwargs)

    t3 = time()
    if print_: print(f'ultra matrix takes {pf(t3-t2,2)}. {pf((t3-t2)/(t2-t1), 2)} times of mst')

    stairs_slice = subslice(d, f_sort, f, idx_sort, mst, sigma, new2old_dict, old2new_dict)
    t4 = time()
    if print_: print(f'subslice matrix takes {pf(t4-t3, 2)}')

    if verbose == 0:
        print('.', end='')
    else:
        if idx % step == 0:
            print(f' {idx}: 1) get_subgraph {pf(t1-t0,2)} 2) mst {pf(t2-t1,2)} '
              f'3) ultra matrix {pf(t3-t2,2)} 4) subslice {pf(t4-t3,2)}')

    return stairs_slice


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--d", default=2, type=int, help='dimension')
parser.add_argument("--name", default='moon', type=str, help='data name')
parser.add_argument("--n_jobs", default=-1, type=int, help='num of jobs')
parser.add_argument("--yscale", default='linear', type=str, help='yscale', choices=['log', 'linear', 'symlog'])
parser.add_argument("--noise", default='None', type=str, help='Noise type', choices=['None', 'uni'])
parser.add_argument("--n_noise", default=0, type=int, help='num of noise points')
parser.add_argument("--n_subsets", default=10, type=int, help='num of subsets of staircases')
parser.add_argument("--bw", default=0.5, type=float, help='bandwidth of kde')
parser.add_argument("--metric", default='euclidean', type=str, help='metric for pairwise dist')
parser.add_argument("--n_features", default=2, type=int, help='num of features for make.blobs')
parser.add_argument("--seed", default=2, type=int, help='random seed')
parser.add_argument("--scale", default=1, type=float, help='scale of dataset. used for test woojin\'s conjecture')


parser.add_argument("--check", action='store_true')
parser.add_argument("--viz_sc", action='store_true')
parser.add_argument("--viz_inter", action='store_true', help='interactive viz')
parser.add_argument("--viz_ind", action='store_true', help='visualize individual staricase')
parser.add_argument("--viz_pts", action='store_true')
parser.add_argument("--viz_hist", action='store_true')
parser.add_argument("--flip", action='store_true')
parser.add_argument("--num", action='store_true')
parser.add_argument("--save", action='store_true', help='save individual image')
parser.add_argument("--hide_tickers", action='store_true')

parser.add_argument("--woojin", action='store_true')
parser.add_argument("--switch", action='store_true')
parser.add_argument("--junc", action='store_true')

parser.add_argument("--line_a", default=1.01, type=float, help='y=ax+b')
parser.add_argument("--line_b", default=1, type=float, help='y=ax+b')


if __name__ == '__main__':
    args = parser.parse_args()
    save_flag = False
    verbose = 2
    np.random.seed(42)

    # specify toy models
    # points, distm = pts_on_square(add_noise=True, n_noise=10)
    # points, _, distm = modelnet(idx = 100)
    points, color, distm = toy_dataset(n_sample=args.n, name=args.name, seed=args.seed, metric=args.metric, n_features=args.n_features, scale=args.scale)
    # points, color, distm, f = woojin2(switch=args.switch)
    # points, color, distm, f = woojin3()
    # points, color, distm = non_uniform(n = args.n, noise=False, d=args.d)
    print(f' ave dist for {args.metric} is {np.average(distm)}')
    if args.viz_inter:
        write_pts_for_viz(points, color, name = args.name)

    if args.noise == 'uni':
        xrange = (min(points[:,0]), max(points[:,0]))
        yrange = (min(points[:, 1]), max(points[:, 1]))
        pts_, color_ =  uniform_noise(xrange=xrange, yrange=yrange, size = args.n_noise)
        points = np.concatenate((points, pts_), axis=0)
        color += color_
        distm = pts2distm(points)

    if args.viz_pts: viz_pd(points, show=True, color=None)

    f = - density(points, bw=args.bw).reshape(len(points), 1)
    print('*' * 50)

    if not args.woojin:
        hist_plot(f, show=args.viz_hist)

    f_sort = copy.deepcopy(f).reshape((len(f),)).tolist() # f_sort is a global variable
    f_sort.sort()
    idx_sort = get_idxsort(f, f_sort)
    sigmas = f.reshape(len(f),).tolist()
    sigmas.sort()

    G = set_graph(f.shape[0], f, distm)
    G = get_subgraph(f, 0, distm)

    g = get_subgraph(f, sigmas[-1], distm)
    msts_list = msts(g, f, distm, print_=False, check=False)
    stairs0 = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, sigmas, mst_opt=True, print_=False, step=100) for sigma in sigmas[1:])

    if args.check: # todo: a small bug. ignore for now.
        #  py I_x/noise.py --n 500. diff for key 417.0 where d1 is None and d2 is (1.8749448416668457, 0.019878904777765473)

        for i in range(len(sigmas) - 1):  # sigmas = np.linspace(0, 1, 10)
            G = update_subgraph(G, f, distm, sigmas[i], sigmas[i + 1])

            home = str(Path.home())
            file = os.path.join(home, 'Documents', 'DeepLearning', 'Clustering', 'subgraphs', str(i + 1) + '.pkl')

            t0 = time()
            with open(file, 'wb') as handle:
                pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
                t1 = time()
                print(f'Saving graph of {len(G.nodes())}/{len(G.edges())}. Takes {pf(t1 - t0, 3)}')

        stairs1 = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, sigmas, mst_opt=False, print_=False) for sigma in sigmas[1:])

        assert len(stairs0) == len(stairs1)
        for i in range(len(stairs0)):
            try:
                assert stairs0[i] == stairs1[i]
            except AssertionError:
                print(f'break for {i}')
                d1, d2 = stairs0[i], stairs1[i]
                for k in d1.keys():
                    if d1.get(k) != d2.get(k):
                        print(f'diff for key {k} where d1 is {d1.get(k)} and d2 is {d2.get(k)}') # where stairs0 is {stairs0[i]} and \n stairs1 is {stairs1}
                for k in d2.keys():
                    if d1.get(k) != d2.get(k):
                        print(f'diff for key {k} where d1 is {d1.get(k)} and d2 is {d2.get(k)}') # where stairs0 is {stairs0[i]} and \n stairs1 is {stairs1}

    # expend 2d_dgm
    d = stairs0[-1]
    extra_slice = {}
    for k, (sig, eps) in d.items():
        extra_slice[k] = (sig + 1, eps)
    stairs0.append(extra_slice)

    res = dgm_format(stairs0, filter = True)

    if args.check:
        dgm = []
        for sigma in sigmas[1:]:
            dgm_slice = D_x_slice_clean(f, f_sort, idx_sort, sigma, sigmas)  # [{key: (sigma, epsilon)}, ...] # todo: only works for D_x_slice_clean, not D_x_slice
            dgm.append(dgm_slice)  # dgm is a list of dict of form {x:(sig,eps), ...}
        assert stairs0 == dgm

    if args.viz_sc:
        title = f'{str(args.yscale)} ' + 'for y axis.'
        viz_stairs_(res, alpha=.03, color = color, yscale=args.yscale, title=title, flip=args.flip, choices=None)
        plt.show()

    n_image = min(args.n_subsets, len(stairs0))
    if args.viz_ind:
        title = f'{str(args.yscale)} ' + 'for y axis.' + f' Stair {i}'
        for i in range(n_image):
            viz_stairs_(res, alpha=.2, color=color, yscale=args.yscale, title=title, flip=args.flip, choices=[i],
                        num=args.num, save=args.save, hide_tickers=args.hide_tickers, name = args.name)
            plt.show()

    n_row = math.ceil(np.sqrt(n_image)+1)
    plt.subplot(n_row, n_row, 1)
    plt.subplots_adjust(left=.01, bottom=.01, right=1, top=1, wspace=.001, hspace=.001)

    juncs = {}
    for i in range(n_image):
        plt.subplot(n_row, n_row, i+1)
        title = f'{str(args.yscale)} ' + 'for y axis.' +  f' Stair {i}'
        junc = viz_stairs_(res, alpha=.2, color=color, yscale=args.yscale, title=title, flip=args.flip, choices=[i], num=args.num, save=args.save)
        juncs[i] = junc
    print(juncs)
    print('-'*50)

    # query related
    input = juncs
    different_juncs = []
    for k, v in input.items():
        single_juncs = v['type1'].tolist()
        different_juncs.append(single_juncs)

    for single_juncs in different_juncs:
        print(single_juncs)
        s = staircase()
        s.build_segs_from_juncs(single_juncs)
        s.find_instersect((args.line_a, args.line_b))
        s.plot_segs()
    sys.exit()

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    with open(cur_dir +  '/conjecture/juncs_noise.pickle', 'wb') as f:
        pickle.dump(juncs, f)

