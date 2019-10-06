import sys
import os
import numpy as np
from helper.time import timefunction
from sys import getsizeof
from joblib import delayed, Parallel

np.random.random(42)
import networkx as nx
import time
import pickle
from I_x.I_x_slice import set_graph
from I_x.mst_test import get_ultra_matrix
from helper.time import precision_format as pf
import copy
from profilehooks import profile

TOR = 1e-6

def g_info(g):
    return
    print(nx.info(g))
    fvs = nx.get_node_attributes(g, 'fv')
    print()

@profile
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

    t0 = time.time()
    g_ = copy.deepcopy(g)
    if print_: print(f'deep copy takes {pf(time.time()-t0, 2)}')
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
    idx = sigmas.index(sigma)
    file = os.path.join(os.path.dirname(__file__), '..', 'subgraphs', str(idx) + '.pkl')
    with open(file, 'rb') as handle:
        g = pickle.load(handle)
    if print_: print(f'Loading graph {len(g)}/{len(g.edges())} from {file}')
    return g

def test_graph(n = 100):
    np.random.seed(42)
    g = nx.random_geometric_graph(n, 0.1)
    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random()
    return g

@profile
def subslice(d, f_sort, f, idx_sort, mst, sigma):
    stairs_slice = []  # also (n^2) time #todo: optmizie
    for x_ in mst.nodes():
        idx = f_sort.index(f[int(x_)][0])
        indices_below = idx_sort[:idx + 1]  #
        # indices_below = [i for i in range(n) if f[i][0] < f[x_][0] + TOR] # todo: optimize this

        # d_sub = d[np.ix_(indices_below, indices_below)]
        # epsilon = min(d_sub[indices_below.index(x_)])

        # idx2 = f.index(f[int(x_)])
        d_sub = d[np.ix_([int(x_)], indices_below)]
        epsilon = min(d_sub[0])
        # print(f'epsilon is {epsilon}. epsilon2 is {epsilon2}. d_sub shape is {d_sub.shape}')
        # assert epsilon2 == epsilon
        stairs_slice.append({x_: (sigma, epsilon)})
    return stairs_slice

# @profile
def D_x_slice(f, f_sort, idx_sort, sigma, print_=False):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x_: idx
    :param sigma: sigma considered for now
    :return:
    """

    t0 = time.time()
    n = f.shape[0]

    # case 2 of I_x
    # if f[x] == min(f): return sigma, 0, np.zeros((n, n))

    # G = get_subgraph(f, sigma, distm)
    G = load_subgraph(sigma, sigmas, print_=True)
    # for j in range(100):
    #     try:
    #         print(f'G[1][{j}] is {G[1][j]}')
    #     except:
    #         pass
    # sys.exit()
    t1 = time.time()
    if print_: print(f'get_subgraph takes {pf(t1-t0,2)}')

    mst = nx.minimum_spanning_tree(G, weight='weight')
    t2 = time.time()
    if print_: print(f'mst takes {pf(t2-t1,2)})')

    d = get_ultra_matrix(mst, n = n, fast=True)
    t3 = time.time()
    if print_:
        print(f'ultra matrix takes {pf(t3-t2,2)}. {pf((t3-t2)/(t2-t1), 2)} times of mst')

    # return sigma, d

    # stairs_slice = [] # also (n^2) time #todo: optmizie
    # for x_ in mst.nodes():
    #     idx = f_sort.index(f[int(x_)][0])
    #     indices_below = idx_sort[:idx+1] #
    #     # indices_below = [i for i in range(n) if f[i][0] < f[x_][0] + TOR] # todo: optimize this
    #     d_sub = d[np.ix_(indices_below, indices_below)]
    #     epsilon = min(d_sub[indices_below.index(x_)])
    #     stairs_slice.append({x_: (sigma, epsilon)} )
    stairs_slice = subslice(d, f_sort, f, idx_sort, mst, sigma)
    t4 = time.time()
    if print_: print(f'subslice matrix takes {pf(t4-t3, 2)}')

    print(f' 1) get_subgraph {pf(t1-t0,2)} 2) mst {pf(t2-t1,2)} '
          f'3) ultra matrix {pf(t3-t2,2)} 4) subslice {pf(t4-t3,2)}')


    return stairs_slice

    # for x_ in mst.nodes():
    #     path_dict = nx.single_source_shortest_path(mst, x_)  # [1, 70, 78, 13, 10]
    #     t3 = time.time()
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
    #     t4 = time.time()
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

def dc_pickle(a):
    return pickle.loads(pickle.dumps(a, -1))


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=500, type=int, help='num of points') # (1515, 500) (3026,)
parser.add_argument("--re", action='store_true', help='recompute subgraphs') # (1515, 500) (3026,)


if __name__ == '__main__':
    args = parser.parse_args()
    n = args.n # 2000

    plot_flag = True
    f = np.random.random((n, 1))
    f[10] = 0.1
    distm = np.random.random((n, n))
    distm = distm + distm.T

    sigmas = f.reshape(n,).tolist()
    sigmas.sort()

    load_subgraph(sigmas[11], sigmas, print_=True)

    f_sort = copy.deepcopy(f).reshape((n,)).tolist() # # f_sort is a global variable
    f_sort.sort()

    f_ = f.tolist()
    f_ = [v[0] for v in f_]
    assert type(f_) == list
    assert len(f_) == n
    assert set(f_) == set(f_sort)
    idx_sort = [f_.index(v) for v in f_sort]

    stair = []
    G = set_graph(f.shape[0], f, distm)
    G = get_subgraph(f, .2, distm)
    g_info(G)
    print(f'graph info for {.2}')

    G = set_graph(f.shape[0], f, distm)
    G = get_subgraph(f, 0, distm)

    subgraphs = {0: G}
    subgraphs_ = {}

    # sigmas = np.linspace(0, 1, 10)
    if args.re:
        for i in range(len(sigmas)-1):
            t0 = time.time()
            G  = update_subgraph(G, f, distm, sigmas[i], sigmas[i+1])
            file = os.path.join(os.path.dirname(__file__), '..', 'subgraphs', str(idx) + '.pkl')
            with open(file, 'wb') as handle:
                pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

            t1 = time.time()
            G_ = get_subgraph(f, sigmas[1], distm) # get_subgraph(f, sigmas[i+1], distm)
            t2 = time.time()
            print(f'G/G_ is {len(G.edges())}/{len(G_.edges())} ')
            print(f'{i} iter: update/get_subgraph takes {pf(t1-t0, 3)}/{pf(t2-t1, 3)} ', end=' ')
            subgraphs_[sigmas[i+1]] = G_

            t3 = time.time()
            subgraphs[sigmas[i + 1]] = G # dc_pickle(G) #copy.deepcopy(G)
            print(f' size of subgraphs is {getsizeof(subgraphs)/1000000} mb. size of G is {getsizeof(G)/1000000}', end=' ')
            t4 = time.time()
            print(f'pickle {pf(t4 - t3, 3)} ', end=' ')

        print(f'len of sigmas is {len(sigmas)}')
        print(f'len of subgraphs/_ is {len(subgraphs)}/{len(subgraphs_)}')

        for k in subgraphs_.keys():
            print(f'sigma is {pf(k,2)} and num of edges is {len(subgraphs[k].edges())}/{len(subgraphs_[k].edges())}')

    stairs = []
    # for sigma in sigmas[:10]:
    #     D_x_slice(f, f_sort, idx_sort, sigma, print_=False)
    # sys.exit()

    stairs = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, print_=False) for sigma in sigmas)
    # print(stairs)
    sys.exit()
    for sigma in sigmas:
        stairs_slice = D_x_slice(f, distm, sigma, print_=False)
        print(stairs_slice)
    sys.exit()


    # get the slices of Dgm
    t0 = time.time()
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
    print(f'finish all stairs of len {len(stairs)} takes {time.time()-t0}')
    # for k, v in stairs.items():
    #     print(k, v)


    sys.exit()
    for i, (sig, eps) in enumerate(stair):
        print(i, sig, eps)
        indices_below = [i for i in range(n) if f[i][0] < f[x][0] + TOR]
        d_sub = ds[i][np.ix_(indices_below, indices_below)]
        assert min(d_sub[indices_below.index(x)]) == eps


