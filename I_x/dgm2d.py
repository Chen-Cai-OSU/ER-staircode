import copy
import os
import pickle
import sys
from time import time
from sys import getsizeof

import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from profilehooks import profile

from I_x.I_x_slice import set_graph
from I_x.mst_test import get_ultra_matrix
from helper.time import precision_format as pf
from helper.time import timefunction
from I_x.blank import msts
from I_x.mst_speedup import mst_total_lengh

TOR = 1e-6
EPSILON_DEFAULT = 0
BACKEND = 'multiprocessing'
np.random.random(42)

def g_info(g):
    return
    print(nx.info(g))
    fvs = nx.get_node_attributes(g, 'fv')
    print()

@timefunction
def update_subgraph(g, f, distm, sigma1, sigma2, print_ = True):
    """ if g is nx graph where all nodes are below sigma1,
        this function will include all extra edges btwn sigma1 and sigma2
    """

    idx = np.multiply(f > sigma1 - TOR, f < sigma2 + TOR)

    idx_old = np.multiply(f < sigma1 - TOR, 1)
    filter_m = np.outer(idx, idx) + np.outer(idx, idx_old) + np.outer(idx_old, idx)

    val = np.multiply(distm, filter_m)
    edgelist = np.argwhere(val > 0)  # np.array of shape (_, 2)
    val = val[val > 0]

    n_ = edgelist.shape[0]  # num of edges in sub-complete graph
    assert edgelist.shape[0] == val.shape[0]
    edgelist = np.concatenate((edgelist, val.reshape((n_, 1))), axis=1)  # array of shape (n_, 3)
    assert edgelist.shape[1] == 3

    edgelist = edgelist.tolist() # a list of list of length 3

    # print(f'Before adding {len(edgelist)} edges')
    g_info(g)
    g.add_weighted_edges_from(edgelist)
    g_info(g)
    return g

    t0 = time()
    g_ = copy.deepcopy(g)
    if print_: print(f'deep copy takes {pf(time()-t0, 2)}')
    return g_

# @timefunction
def get_subgraph(f, sigma, distm, print_=False):
    """
    get the subgraph that is below sigma. Right now it is slow when sigma is large.

    :param f: function value for nodes. Array of shape (n, 1)
    :param sigma: threshold
    :param distm: Dist matrix of shape (n, n)
    :param print_: For debug
    :return: A networkx graph
    """

    filter_m = np.outer([f < sigma + TOR], [f < sigma + TOR])  # todo: viz filter_m should be square
    val = np.multiply(distm, filter_m)
    edgelist = np.argwhere(val > 0)  # np.array of shape (_, 2)
    val = val[val > 0]

    n_ = edgelist.shape[0]  # num of edges in sub-complete graph
    assert edgelist.shape[0] == val.shape[0]
    edgelist = np.concatenate((edgelist, val.reshape((n_, 1))), axis=1)  # array of shape (n_, 3)
    assert edgelist.shape[1] == 3

    # format data for nx
    lines = []
    for i in range(n_):
        s, t, weight = edgelist[i][0], edgelist[i][1], edgelist[i][2]
        line = f"{int(s)} {int(t)}" + ' {' + f"'weight':{float(weight)}" + '}'
        lines.append(line)

    G = nx.parse_edgelist(lines, nodetype=int)
    # return G
    if print_:
        print(f'G info in slice_ {nx.info(G)}')
        nodes = list(G.nodes())
        nodes.sort()
        print(f'Nodes of G in slices_ is {nodes}')
    return G

def load_subgraph(sigma, sigmas, print_ = False):
    """
    :param sigma:
    :param sigmas:
    :param print_:  for debug
    :return: return a nx graph where nodes are below sigma
    """
    t0 = time()
    idx = sigmas.index(sigma)

    file = os.path.join(os.path.dirname(__file__), '..', 'subgraphs', str(idx) + '.pkl')

    with open(file, 'rb') as handle:
        g = pickle.load(handle)
    t1 = time()

    if print_: print(f'Loading graph {len(g)}/{len(g.edges())} from {file}. Takes {pf(t1-t0, 3)}')
    return g

def test_graph(n = 100):
    np.random.seed(42)
    g = nx.random_geometric_graph(n, 0.1)
    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random()
    return g

def subslice(d, f_sort, f, idx_sort, mst, sigma):
    """

    :param d: ultra metric matrix of shape (n, n)
    :param f_sort: sorted filtration function
    :param f: original filtration function
    :param idx_sort: the index idx of each element in f so that all larger ones can be found by f_sort[idx:]
    :param mst: minimal spanning tree
    :param sigma: slice
    :return:
    """

    # return sigma, d

    # stairs_slice = [] # also (n^2) time #todo: optmizie
    # for x_ in mst.nodes():
    #     idx = f_sort.index(f[int(x_)][0])
    #     indices_below = idx_sort[:idx+1] #
    #     # indices_below = [i for i in range(n) if f[i][0] < f[x_][0] + TOR] # todo: optimize this
    #     d_sub = d[np.ix_(indices_below, indices_below)]
    #     epsilon = min(d_sub[indices_below.index(x_)])
    #     stairs_slice.append({x_: (sigma, epsilon)} )

    stairs_slice = {}  # also (n^2) time #todo: optmizie
    for x_ in mst.nodes():
        idx = f_sort.index(f[int(x_)][0])
        indices_below = idx_sort[:idx + 1]  #
        d_sub = d[np.ix_([int(x_)], indices_below)]
        epsilon = min(d_sub[0])
        if epsilon == 1e5:
            assert len(d_sub[0])==1
            epsilon = EPSILON_DEFAULT
        stairs_slice[x_] = (sigma, epsilon)
    return stairs_slice

# @profile
def D_x_slice(f, f_sort, idx_sort, sigma, print_=False, mst_opt = False, verbose = 1, **kwargs):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x_: idx
    :param sigma: sigma considered for now
    :return: A slice of 2d diagram (a list of dict of form {x_: (sigma, epsilon)} )
    """

    assert 'sigmas' in globals()

    t0 = time()
    n = f.shape[0]
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
    if print_: print(f'mst takes {pf(t2 - t1, 2)})')

    d = get_ultra_matrix(mst, n = n, fast=True, faster=False)
    t3 = time()
    if print_: print(f'ultra matrix takes {pf(t3-t2,2)}. {pf((t3-t2)/(t2-t1), 2)} times of mst')

    stairs_slice = subslice(d, f_sort, f, idx_sort, mst, sigma)
    t4 = time()
    if print_: print(f'subslice matrix takes {pf(t4-t3, 2)}')

    if verbose == 0:
        print('.', end='')
    else:
        print(f' {idx}: 1) get_subgraph {pf(t1-t0,2)} 2) mst {pf(t2-t1,2)} '
          f'3) ultra matrix {pf(t3-t2,2)} 4) subslice {pf(t4-t3,2)}')

    return stairs_slice

    # for x_ in mst.nodes():
    #     path_dict = nx.single_source_shortest_path(mst, x_)  # [1, 70, 78, 13, 10]
    #     t3 = time()
    #
    #     if f[x_] > sigma: continue
    #     indices_below = [i for i in range(n) if f[i][0] < f[x_][0] + TOR]
    #
    #     pathkeys = path_dict.keys()
    #     path_dict_filter = dict((k, path_dict[k]) for k in indices_below if k in pathkeys)  # filter path_dict
    #     length_dict = {}
    #
    #     for k, v in path_dict_filter.items():  # v is like [10, 7, 0]
    #         if len(v) > 1:
    #             lenlist = [mst[v[i]][v[i + 1]]['weight'] for i in range(len(v) - 1)]
    #         else:
    #             lenlist = [1e5]
    #         length_dict[k] = lenlist
    #     t4 = time()
    #     print(f'get length_dict takes {t4 - t2}')
    #
    #     assert f[x_] <= sigma
    #     d_sub = d[np.ix_(indices_below, indices_below)]
    #     epsilon = min(list(map(max, length_dict.values())))
    #
    #     print(min(d_sub[indices_below.index(x_)]), epsilon)
    #     print('d_sub    ', sorted(d_sub[indices_below.index(x_)]))
    #     print('original ', sorted(list(map(max, length_dict.values()))))
    #
    #     assert min(d_sub[indices_below.index(x_)]) == epsilon

def D_x_slice_clean(f, f_sort, idx_sort, sigma, sigmas, mst_opt = False):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x_: idx
    :param sigma: sigma considered for now
    :return: A slice of 2d diagram (a list of dict of form {x_: (sigma, epsilon)} )
    """

    # assert 'sigmas' in globals()

    n = f.shape[0]
    if mst_opt:
        idx = sigmas.index(sigma)
        mst = msts_list[idx]
    else:
        G = load_subgraph(sigma, sigmas, print_=False)
        mst = nx.minimum_spanning_tree(G, weight='weight')

    d = get_ultra_matrix(mst, n = n, fast=True)

    stairs_slice = subslice(d, f_sort, f, idx_sort, mst, sigma)

    return stairs_slice

def dgm_format(dgm, filter = False):
    """
    used for viz_stairs_
    :param dgm: dgm is a list of dict of form {x:(sig,eps), ...}
    :param filter: filter out
    :return: a list of lists of tuples
    """
    stairs = {} # a dict of form (x: [(sig1, eps1), (sig2, eps2), ...])
    for s in dgm:
        for x, (sig, eps) in s.items():
            if x not in stairs.keys(): stairs[x] = []
            stairs[x].append((sig, eps))

    if filter:
        res = []
        for k, v in stairs.items():
            for (sig, ep) in v:
                if ep == 1e5:
                    print(f'filter out node {k} due to 1e5')
                    break
            res.append(v)
    else:
        res = list(stairs.values())
    return res

def dc_pickle(a):
    return pickle.loads(pickle.dumps(a, -1))

def get_idxsort(f, f_sort):
    """
    :param f: array
    :param f_sort: a list
    :return: the index idx of each element in f so that all larger ones can be found by f_sort[idx:]
    """
    f_ = f.tolist()
    f_ = [v[0] for v in f_]
    assert type(f_) == list
    assert set(f_) == set(f_sort)
    idx_sort = [f_.index(v) for v in f_sort]
    return idx_sort

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points') # (1515, 500) (3026,)
parser.add_argument("--n_jobs", default=1, type=int, help='num of jobs') # (1515, 500) (3026,)
parser.add_argument("--re", action='store_true', help='recompute subgraphs') # (1515, 500) (3026,)
parser.add_argument("--re", action='store_true', help='efficiently computing all msts') # (1515, 500) (3026,)


def set_dir():
    dir = os.path.join(os.path.dirname(__file__), '..', 'subgraphs')
    from helper.io_related import make_dir
    make_dir(dir)

if __name__ == '__main__':
    np.random.seed(42)
    set_dir()
    args = parser.parse_args()
    n = args.n # 2000

    f = np.random.random((n, 1))
    f[3] = 0.1
    distm = np.random.random((n, n))
    distm = distm + distm.T

    sigmas = f.reshape(n,).tolist()
    sigmas.sort()

    f_sort = copy.deepcopy(f).reshape((n,)).tolist() # f_sort is a global variable
    f_sort.sort()
    idx_sort = get_idxsort(f, f_sort)

    G = set_graph(f.shape[0], f, distm)
    G = get_subgraph(f, 0, distm)

    subgraphs = {0: G}
    subgraphs_ = {}
    if args.re:
        for i in range(len(sigmas)-1): # sigmas = np.linspace(0, 1, 10)
            t0 = time()
            G  = update_subgraph(G, f, distm, sigmas[i], sigmas[i+1])
            file = os.path.join(os.path.dirname(__file__), '..', 'subgraphs', str(i+1) + '.pkl')
            with open(file, 'wb') as handle:
                pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

            t1 = time()
            G_ = get_subgraph(f, sigmas[1], distm) # get_subgraph(f, sigmas[i+1], distm)
            t2 = time()
            print(f'G/G_ is {len(G.edges())}/{len(G_.edges())} ')
            print(f'{i} iter: update/get_subgraph takes {pf(t1-t0, 3)}/{pf(t2-t1, 3)}. ', end=' ')
            subgraphs_[sigmas[i+1]] = G_

            t3 = time()
            subgraphs[sigmas[i + 1]] = G
            t4 = time()
            print(f'pickle {pf(t4 - t3, 3)} ', end=' ')

        print(f'len of sigmas is {len(sigmas)}')
        print(f'len of subgraphs/_ is {len(subgraphs)}/{len(subgraphs_)}.')

        for k in subgraphs_.keys():
            print(f'sigma is {pf(k,2)} and num of edges is {len(subgraphs[k].edges())}/{len(subgraphs_[k].edges())}.')
        print('-'*150)

    # for sigma in sigmas[-10:]:
    #     D_x_slice(f, f_sort, idx_sort, sigma, print_=False)
    # sys.exit()
    # stairs = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(D_x_slice_clean)(f, f_sort, idx_sort, sigma) for sigma in sigmas[-120:])

    g = get_subgraph(f, sigmas[-1], distm)

    msts_list = msts(g, f, distm, print_=False, check=False)

    stairs0 = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, mst_opt = True, print_=False) for sigma in sigmas[-8:])
    stairs0 = stairs0[0]
    print('-'*150)
    sys.exit()

    stairs = Parallel(n_jobs=args.n_jobs, backend=BACKEND)(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, print_=False) for sigma in sigmas[-20:])
    stairs = stairs[0]

    for k, v in stairs0.items():
        if v != stairs.get(k, None):
            print(k, v, stairs.get(k, None))



    sys.exit()
    for sigma in sigmas:
        stairs_slice = D_x_slice(f, distm, sigma, print_=False)
        print(stairs_slice)
    sys.exit()


    # get the slices of Dgm
    t0 = time()
    stairs = {}
    for x_ in range(n): # different x
        stair_x_ = []
        indices_below = [i for i in range(n) if f[i][0] < f[x_][0] + TOR]
        for i in range(len(ds)): # different sigmas
            sig = ds[i][0]
            d_sub = ds[i][1][np.ix_(indices_below, indices_below)]
            eps_x_ = min(d_sub[indices_below.index(x_)])
            stair_x_.append((sig, eps_x_))
        stairs[x_] = stair_x_
    print(f'finish all stairs of len {len(stairs)} takes {time()-t0}')
    # for k, v in stairs.items():
    #     print(k, v)


    sys.exit()
    for i, (sig, eps) in enumerate(stair):
        print(i, sig, eps)
        indices_below = [i for i in range(n) if f[i][0] < f[x][0] + TOR]
        d_sub = ds[i][np.ix_(indices_below, indices_below)]
        assert min(d_sub[indices_below.index(x)]) == eps


