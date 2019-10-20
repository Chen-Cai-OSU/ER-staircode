import numpy as np
import networkx as nx
from I_x.mst_speedup import mst_test, range_

def msts(g, print_ = False, check = False):
    n = len(g)
    for u, v in g.edges():
        g[u][v]['weight'] = np.random.random()  # dist_(g.node[u]['pos'], g.node[v]['pos'])
    for u in g.nodes():
        g.node[u]['fv'] = np.random.random()

    nv_dict = nx.get_node_attributes(g, 'fv')
    sorted_x = sorted(nv_dict.items(), key=lambda kv: kv[1])
    node_list = [x[0] for x in sorted_x]

    g0 = nx.subgraph(g, range_(node_list, 1))
    mst = nx.minimum_spanning_tree(g0, weight='weight')

    msts = [mst]
    for idx in range(n):  # range_(node_list, n-1):
        mst = mst_test(mst, g, idx, node_list, print_=print_, check=check)
        msts.append(mst)
    return msts[:-1]

if __name__ == '__main__':
    n = 1000
    g = nx.complete_graph(n)
    msts = msts(g, print_=False, check=False)
    # print(list(map(len, msts)))
    assert list(map(len, msts)) == list(range(1, n+1))