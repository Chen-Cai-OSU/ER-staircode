from time import time
import copy
import numpy as np
from helper.time import precision_format as pf
np.random.random(42)
import networkx as nx
import sys

def mst_total_lengh(mst, print_ = True):
    assert nx.is_connected(mst)
    assert nx.is_tree(mst)
    mst_length = np.sum(list(nx.get_edge_attributes(mst, 'weight').values()))
    if print_:
        # tmp = list(mst.nodes())
        # tmp.sort()
        # print(tmp, end=' ')
        print(mst_length)
    return mst_length

# def mst_test(g, g_ori, mst, idx, print=False):
#     """
#     :param g: original graph below sigma
#     :param g_ori: original graph
#     :param mst: mst below sigma
#     :param idx: the node idx needs to be added
#     :return:
#     """
#     pass
#     add_idx = idx
#
#     t1 = time()
#     mst_total_lengh(mst)
#     t2 = time()
#     if print: print(f'ori method takes {pf(t2 - t1, 3)}')
#
#     mst_plus = nx.minimum_spanning_tree(g, weight='weight')
#     mst_plus.add_weighted_edges_from(g_ori.edges(add_idx, data='weight'))
#
#     t1 = time()
#     mst_total_lengh(nx.minimum_spanning_tree(mst_plus, weight='weight'))
#     t2 = time()
#     if print: print(f'modified method takes {pf(t2 - t1, 3)}')
#
#     return nx.minimum_spanning_tree(mst_plus), g.add_weighted_edges_from(g_ori.edges(add_idx, data='weight'))

def mst_test(mst_i, g_ori, idx, lis, print_=False, check = True):
    """
    :param mst_i: including idx i
    :param g_ori: original graph
    :param g: g_{i+1}
    :return: mst_{i+1}
    """
    if idx + 1 > len(lis):
        g = g_ori
    else:
        g = nx.subgraph(g_ori, range_(lis, idx+1))

    if check:
        t1 = time()
        mst1 = nx.minimum_spanning_tree(g) # mst_{i+1}
        t2 = time()
        if print_: print(f'ori method takes {pf(t2 - t1, 3)}')
        mst_total_lengh(mst1, print_=print_)

    t1 = time()

    mst_i.add_weighted_edges_from(g.edges(lis[idx], data='weight'))
    mst2 = nx.minimum_spanning_tree(mst_i)
    t2 = time()
    if print_: print(f'new method takes {pf(t2 - t1, 3)}')

    mst_total_lengh(mst2, print_=print_)
    if print_: print('-'*20)

    return mst2

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=50, type=int, help='num of nodes') # (1515, 500) (3026,)

def range_(lis, i):
    # used to replace range(i) with range_(lis, i)
    return lis[:i]

if __name__ == '__main__':
    # g = nx.random_geometric_graph(100, 0.8)
    args = parser.parse_args()
    n = args.n
    g = nx.complete_graph(n)

    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random()  # dist_(g.node[u]['pos'], g.node[v]['pos'])
    for u in g.nodes():
        g.node[u]['fv'] = np.random.random()

    nv_dict = nx.get_node_attributes(g, 'fv')
    sorted_x = sorted(nv_dict.items(), key=lambda kv: kv[1])
    node_list = [x[0] for x in sorted_x]

    g0 = nx.subgraph(g, range_(node_list, 1))
    mst = nx.minimum_spanning_tree(g0, weight='weight')

    msts = []
    for idx in range(n-1): #range_(node_list, n-1):
        mst = mst_test(mst, g, idx, node_list, print_=False, check=True)
        msts.append(mst)

