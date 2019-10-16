import sys
from time import time
import numpy as np
from helper.time import timefunction
from helper.format import precision_format as pf
from profilehooks import profile

np.random.random(42)
import networkx as nx

TOR = 1e-6

@timefunction
def get_subgraph(f, sigma, distm, print_=False):

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

def test_graph(n = 100, tree = False, print_ = False):
    np.random.seed(42)
    if tree:
        g = nx.random_tree(n, seed=42)
        # g = nx.balanced_tree(2,3)
    else:
        g = nx.random_geometric_graph(n, 0.3, seed=42)
    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random()
        if print_:
            print(f' edge {u}-{v} has weight {g[u][v]}')
    return g

def I_x_slice_(f, distm, x, sigma, print_=False):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x: idx
    :param sigma: sigma considered for now
    :return:
    """

    t0 = time.time()
    n = f.shape[0]

    # case 2 of I_x
    if f[x] == min(f): return sigma, 0, np.zeros((n,n))

    indices_below = [i for i in range(n) if f[i][0] < f[x][0] + TOR]
    assert f[x] <= sigma

    G = get_subgraph(f, sigma, distm)
    # G = test_graph(n=n)
    t1 = time.time()
    print(f'get_subgraph takes {t1-t0}')

    mst = nx.minimum_spanning_tree(G, weight='weight')
    t2 = time.time()
    print(f'mst takes {t2-t1}')


    path_dict = nx.single_source_shortest_path(mst, x)  # [1, 70, 78, 13, 10]
    t3 = time.time()
    print(f'single source shortest path takes {t3 - t2}')

    # print(f'path dict is {path_dict}\n')

    pathkeys = path_dict.keys()
    path_dict_filter = dict((k, path_dict[k]) for k in indices_below if k in pathkeys) # filter path_dict
    length_dict = {}


    for k, v in path_dict_filter.items():  # v is like [10, 7, 0]
        if len(v) > 1:
            lenlist = [mst[v[i]][v[i + 1]]['weight'] for i in range(len(v) - 1)]
        else:
            # return sigma, -2
            lenlist = [1e10]
        length_dict[k] = lenlist
    t4 = time.time()
    print(f'get length_dict takes {t4-t2}')


    d = np.zeros((n, n))
    for k, v in length_dict.items():
        d[x][k] = max(v)
    x_row = d[x]

    d1 = np.zeros((n,n)) + x_row.reshape((1, n))
    d2 = np.zeros((n,n)) + x_row.reshape((n, 1))
    d = np.maximum(d1, d2)
    d[x] = x_row

    # for i in list(range(x)) + list(range(x+1, n)): #range(n):
    #     for j in range(n):
    #         if i!=x and j not in path_dict.keys():
    #             d[i][j] = max(d[x][i], d[x][j])

    t5 = time.time()
    print(f'ultra matrix takes {t5-t4}')
    # print(f'ultra metric matrix at row {x} is {d[x]}')

    d_sub = d[np.ix_(indices_below, indices_below)]
    # print(indices_below)
    # print(d_sub.shape)


    epsilon = min(list(map(max, length_dict.values())))
    assert min(d_sub[indices_below.index(x)]) == epsilon


    return sigma, epsilon, d

# @profile
def bfs(g, source, ultrametric_dict, visited):
    if len(visited)==len(g):
        return ultrametric_dict, visited

    nbrs = nx.neighbors(g, source)
    for n in nbrs:
        if n in visited: continue
        ultrametric_dict[n] = max(g[source][n]['weight'], ultrametric_dict.get(source, -1)) # todo has problem
        visited.append(n)
        ultrametric_dict, visited = bfs(g, n, ultrametric_dict, visited)

    return ultrametric_dict, visited

from numba import jit

# @profile
# @jit(nopython=True)
def bfs_fast_test(g, source):
    iter = nx.bfs_successors(g, source)
    ultrametric_dict = {source: 0}

    def hack(n, nbrs):
        tmp = ultrametric_dict[n]
        for nbr in nbrs:
            ultrametric_dict[nbr] = max(g[n][nbr]['weight'], tmp)

    # ultrametric = []
    iter = list(iter)
    [hack(n, nbrs) for n, nbrs in iter]
    return ultrametric_dict

# iter = nx.bfs_successors(g, source)
# attr = nx.get_edge_attributes(g, 'weight')

@jit(nopython=True)
def bfs_fast_test_(iter, source, attr):
    n_node = attr.shape[0]
    ultrametric_dict = np.zeros(n_node)

    def hack(n, nbrs):
        tmp = ultrametric_dict[n]
        for nbr in nbrs:
            ultrametric_dict[nbr] = max(attr[(n, nbr)], tmp, attr[(nbr, n)])
            # ultrametric_dict[nbr] = max(attr.get((n, nbr), -1), tmp, attr.get((nbr, n), -1))

    iter = list(iter)
    [hack(n, nbrs) for n, nbrs in iter]
    return ultrametric_dict


def bfs_fast(g, source, dummy= False):
    iter = nx.bfs_successors(g, source)
    ultrametric_dict = {source: 0}
    if dummy:
        for v in g.nodes():
            ultrametric_dict[v] = 0
        return ultrametric_dict

    for n, nbrs in iter:
        tmp = ultrametric_dict[n]
        g_n = g[n]
        for nbr in nbrs:
            ultrametric_dict[nbr] = max(g_n[nbr]['weight'], tmp)

    # for n, nbrs in iter:
    #     tmp = ultrametric_dict[n]
    #     tmp_dic = dict([(nbr, max(g[n][nbr]['weight'], tmp)) for nbr in nbrs])
    #     ultrametric_dict.update(tmp_dic)
        # ultrametric_dict = {**ultrametric_dict, **tmp_dic}

    return ultrametric_dict

from joblib import delayed, Parallel
from bfs_fast import  bfs_fast_complied, bfs_fast_complied2, bfs_fast_complied3, bfs_fast_complied4, bfs_fast_complied5, bfs_fast_complied6

def get_ultra_matrix(g, n = 100, fast=True):
    """
    :param g: nx tree! with weight attribute for each edge
    :param n: number of nodes
    :param fast: True by default
    :return: ultra_metrix matrix of shape (n, n)
    """
    assert nx.is_tree(g)
    ultra_matrix = np.zeros((n, n)) - 12345
    nodes = g.nodes()

    attr_dict = nx.get_edge_attributes(g, 'weight')

    if fast:
        for source in nodes:
            source = int(source)
            # t0 = time.time()

            # iter = nx.bfs_successors(g, source)
            # iter = list(iter)
            # attr = nx.get_edge_attributes(g, 'weight')
            # attr = np.zeros((len(g), len(g)))
            # ultrametric_dict = bfs_fast_test_(iter, source, attr)

            # ultrametric_dict = bfs_fast_test(g, source)
            # ultrametric_dict = bfs_fast(g, source, dummy=False)

            iter = nx.bfs_successors(g, source)
            iter = list(iter)
            ultrametric_dict = bfs_fast_complied3(attr_dict, source, iter)
            # assert ultrametric_dict_ == ultrametric_dict
            # t4 = time.time()

            # t1 = time.time()
            for k, v in ultrametric_dict.items():
                ultra_matrix[source, int(k)] = v
            # t2 = time.time()
            # print(f'compute ultrametrix_dict takes {pf((t1 - t0)/(t2-t1), 3)} time.')

    else:
        for source in nodes:
            visited = [source]
            ultrametric_dict = {source: 0}
            ultrametric_dict, _ = bfs(g, source, ultrametric_dict, visited)
            for k, v in ultrametric_dict.items():
                ultra_matrix[int(source)][int(k)] = v

    assert (ultra_matrix == ultra_matrix.T).all()
    ultra_matrix += 1e5 * np.eye(n)
    return ultra_matrix

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=50, type=int, help='num of points') # (1515, 500) (3026,)


def mst_total_lengh(mst):
    assert nx.is_connected(mst)
    assert nx.is_tree(mst)
    mst_length = np.sum(list(nx.get_edge_attributes(mst, 'weight').values()))
    print(mst_length)
    return mst_length

def dist_(l1, l2):
    # euclidean distance btwn two lists
    l1, l2 = np.array(l1), np.array(l2)
    from numpy import linalg as LA
    return LA.norm(l1-l2, ord=None)


if __name__ == '__main__':
    l1 = np.random.random(2)
    l2 = np.random.random(2)

    import copy
    np.random.seed(42)
    # g = nx.random_tree(1000, seed=42)
    g = nx.random_geometric_graph(1000, 0.5)
    remove_idx = 10
    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random() # dist_(g.node[u]['pos'], g.node[v]['pos'])

    t1 = time()
    mst_total_lengh(nx.minimum_spanning_tree(g, weight='weight'))
    t2 = time()
    print(f'ori method takes {t2 - t1}')

    g_ = copy.deepcopy(g)
    g_.remove_node(remove_idx)

    mst = nx.minimum_spanning_tree(g_, weight='weight')
    mst.add_weighted_edges_from(g.edges(remove_idx, data='weight'))

    t1 = time()
    mst_total_lengh(nx.minimum_spanning_tree(mst, weight='weight'))
    t2 = time()
    print(f'modified method takes {t2-t1}')
    sys.exit()


    mst_ = nx.minimum_spanning_tree(g, weight='weight')
    length2 = mst_total_lengh(mst_)

    try:
        assert length2 + delta == length1
    except:
        print(f'diff is {length2 + delta - length1}')
    print(f'delta is {delta}')


    source = 1

    t0 = time.time()




    sys.exit()
    ultrametric_dict_fast = bfs_fast(g, source)
    t1 = time.time()
    print(f'bfs_fast takes {pf(t1 - t0, 2)}')


    t0 = time.time()
    visited = [source]
    ultrametric_dict = {source: 0}
    ultrametric_dict, _ = bfs(g, source, ultrametric_dict, visited)

    t1 = time.time()
    print(f'bfs takes {pf(t1 - t0, 2)}')

    assert ultrametric_dict == ultrametric_dict_fast
    sys.exit()

    args = parser.parse_args()
    n = args.n # 2000
    g = test_graph(n=n, tree=True)

    ultra_matrix = np.zeros((n, n))
    for source in g.nodes():
        visited = [source]
        ultrametric_dict = {source: 0}
        ultrametric_dict, _ = bfs(g, source, ultrametric_dict, visited)
        for k, v in ultrametric_dict.items():
            ultra_matrix[source][k] = v
    assert (ultra_matrix == ultra_matrix.T).all()
    ultra_matrix += 1e5 * np.eye(n)

    # ultrametric_dict = {9:0}
    # visited = [9]
    # ultrametric_dict, _ = bfs(g, 9, ultrametric_dict, visited)
    # print(ultrametric_dict[0])

    sys.exit()

    f = np.random.random((n, 1))
    f[10] = 0.1
    distm = np.random.random((n, n))
    distm = distm + distm.T
    x = 10
    sigma = 0.2
    sigmas = [f[i][0] for i in range(n) if f[i][0] > sigma]
    sigmas.sort()
    plot_flag = True

    source, sink = 10, 14
    g = test_graph(n=n)
    for algo in ['prim', 'kruskal', 'boruvka']:
        mst = nx.minimum_spanning_tree(g, weight='weight', algorithm=algo)
        len_list = nx.single_source_dijkstra_path(mst, source, weight='weight')[sink]
        bd_dist = max([mst[len_list[i]][len_list[i+1]]['weight'] for i in range(len(len_list)-1)])
        print(f'algorithm {algo} from {source} to {sink} bottleneck edge weight is {bd_dist}')
    sys.exit()

    source, sink = 10, 18
    g = test_graph(n=n)
    mst = nx.minimum_spanning_tree(g, weight='weight')
    len_list = nx.single_source_dijkstra_path(mst, source, weight='weight')[sink]
    bd_dist = max([mst[len_list[i]][len_list[i + 1]]['weight'] for i in range(len(len_list) - 1)])
    print(f'from {source} to {sink} bottleneck edge weight is {bd_dist}')

    source, sink = 18, 14
    g = test_graph(n=n)
    mst = nx.minimum_spanning_tree(g, weight='weight')
    len_list = nx.single_source_dijkstra_path(mst, source, weight='weight')[sink]
    bd_dist = max([mst[len_list[i]][len_list[i + 1]]['weight'] for i in range(len(len_list) - 1)])
    print(f'from {source} to {sink} bottleneck edge weight is {bd_dist}')
