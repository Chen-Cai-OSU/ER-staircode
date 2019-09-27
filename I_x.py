import matplotlib.pyplot as plt
import numpy as np
import sys
import networkx as nx
from util import viz_stair
np.random.random(42)
import networkx as nx
from Esme.helper.time import timefunction
TOR = 1e-6


def expand_list(l):
    """
    :param l: a list of lists
    :return: a expanded list
    """
    res = []
    for lis in l:
        res += lis
    return res

@timefunction
def set_graph(n, f, distm):
    """
    :param n: graph size
    :param f: np.array of shape (n, 1)
    :param distm: np.array of shape (n, n)
    :return: a complete graph G with G[n]['fv'] = f[n][0] and G[u][v]['w'] = distm[u][v]
    """

    assert f.shape[0] == distm.shape[0] == n
    assert (distm == distm.T).all()
    G = nx.complete_graph(n)
    add_edges = [(i,i) for i in range(n)]
    G.add_edges_from(add_edges)
    assert len(G.edges) == n*(n-1)* 0.5 + n

    edgeval_dict, node_val_dict = {}, {}

    for u, v in G.edges():
        edgeval_dict[(u, v)] = distm[u][v]
    nx.set_edge_attributes(G, edgeval_dict, 'weight')

    for n in G.nodes():
        node_val_dict[n] = f[n][0]
    nx.set_node_attributes(G, node_val_dict, name='fv')
    return G

@timefunction
def G_sub(G, sigma = 0.3, test = False):
    """
    :param G: a graph with fv on node
    :param sigma: filter threshold
    :return: a subgraph of G where only nodes with values below sigma are included
    """
    
    if test:
        n = 100
        G = nx.random_geometric_graph(n, 0.1)
        edgeval_dict = {}
        for u, v in G.edges():
            edgeval_dict[(u,v)] = np.random.random()
        nx.set_edge_attributes(G, edgeval_dict, 'w')

        node_val_dict = {}
        for n in G.nodes():
            node_val_dict[n] = np.random.random()
        nx.set_node_attributes(G, node_val_dict, name='fv')

    n = len(G)
    indices_below = [i for i in range(n) if G.node[i]['fv'] < sigma + TOR]
    G_ = nx.subgraph(G, indices_below)
    return G_

@timefunction
def I_x_slice(f, distm, x, sigma, print_=False, complete_G = None):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x: idx
    :param sigma: sigma considered for now
    :return:
    """

    assert f[x] <= sigma
    # G = set_graph(f.shape[0], f, distm)
    G = complete_G
    G = G_sub(G, sigma=sigma)

    if print_:
        print(f'G info in slice {nx.info(G)}')
        print(f'Nodes of G in slices is {list(G.nodes())}')


    mst = nx.minimum_spanning_tree(G)

    path_dict = nx.single_source_shortest_path(mst, x)  # [1, 70, 78, 13, 10]
    indices_below = [i for i in range(n) if f[i][0] < f[x][0] + TOR]
    pathkeys = path_dict.keys()
    path_dict = dict((k, path_dict[k]) for k in indices_below if k in pathkeys)
    length_dict = {}

    # todo try to use single_path_length
    for k, v in path_dict.items(): # v is like [10, 7, 0]
        if len(v) > 1:
            lenlist = [mst[v[i]][v[i+1]]['weight'] for i in range(len(v)-1)]
        else:
            lenlist = [1e10]
        length_dict[k] = lenlist

    # epsion = min(expand_list(list(length_dict.values())))
    epsilon = min(list(map(max, length_dict.values())))

    return sigma, epsilon

# @timefunction
def I_x_slice_(f, distm, x, sigma, print_=False):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x: idx
    :param sigma: sigma considered for now
    :return:
    """

    n = f.shape[0]
    # indices_below = [i for i in range(n) if f[i][0] < sigma + TOR]
    indices_below = [i for i in range(n) if f[i][0] < f[x][0] + TOR]
    assert f[x] <= sigma

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

    mst = nx.minimum_spanning_tree(G)
    path_dict = nx.single_source_shortest_path(mst, x)  # [1, 70, 78, 13, 10]
    pathkeys = path_dict.keys()
    path_dict = dict((k, path_dict[k]) for k in indices_below if k in pathkeys)
    length_dict = {}

    # todo try to use single_path_length
    for k, v in path_dict.items(): # v is like [10, 7, 0]
        if len(v) > 1:
            lenlist = [mst[v[i]][v[i+1]]['weight'] for i in range(len(v)-1)]
        else:
            lenlist = [1e10]
        length_dict[k] = lenlist

    epsilon = min(list(map(max, length_dict.values())))
    # epsion = min(expand_list(list(length_dict.values())))

    return sigma, epsilon


if __name__ == '__main__':
    n = 100
    f = np.random.random((n, 1))
    f[10] = 0.1
    distm = np.random.random((n, n))
    distm = distm + distm.T
    x = 10
    sigma = 0.2
    sigmas = [f[i][0] for i in range(n) if f[i][0] > sigma]
    sigmas.sort()
    plot_flag = True
    # sigmas = sigmas[20:23]

    stair = []
    G = set_graph(f.shape[0], f, distm)
    for sigma in sigmas:
        sig, eps = I_x_slice(f, distm, x, sigma, complete_G=G, print_=False)
        stair.append((sig, eps))
    viz_stair(stair, plot=plot_flag)
    print('-'*100)

    stair = []
    for sigma in sigmas:
        sig, eps = I_x_slice_(f, distm, x, sigma, print_=False)
        stair.append((sig, eps))
    viz_stair(stair, plot=plot_flag, title='slice_')





