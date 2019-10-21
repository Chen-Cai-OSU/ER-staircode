import networkx as nx
import numpy as np
import itertools

from bfs_fast import  bfs_fast_complied, bfs_fast_complied2, bfs_fast_complied3, bfs_fast_complied4, bfs_fast_complied5, bfs_fast_complied6
from example_py_cy import primes_python_complied
from example import primes
from time import time
import sys
from helper.format import precision_format as pf
from bfs_faster import bfs_faster, bfs_faster2


def bfs_fast_python(g, source):
    iter = nx.bfs_successors(g, source)
    ultrametric_dict = {source: 0}

    for n, nbrs in iter:
        tmp = ultrametric_dict[n]
        g_n = g[n]
        for nbr in nbrs:
            ultrametric_dict[nbr] = max(g_n[nbr]['weight'], tmp)

    return ultrametric_dict

def primes_python_(nb_primes):
    p = []
    n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

def bfs_successors(G, source, depth_limit=None):
    for p, c in nx.bfs_edges(G, source, depth_limit=depth_limit):
        yield (p, c)

def get_iter(g, source, print_ = False, nx_flag = True):
    # get iter_array efficiently
    t0 = time()
    if nx_flag:
        iter = list(nx.bfs_successors(g, source))
        iter_list = [[v] + nbrs for v, nbrs in iter]
        iter_array = np.array(list(itertools.zip_longest(*iter_list, fillvalue=0))).T

        iter_array = np.concatenate([iter_array, np.array(list(map(len, iter_list))).reshape(-1,1)], axis=1) # every row is of form [512 474   0 ...   0   0   2]
        iter_array = iter_array.astype(int)
    else:
        iter_array = np.array(list((bfs_successors(g, source))))
        # print(iter_array)

    t1 = time()
    if print_:
        print(iter_array)
        print(f'get_iter for graph of size {len(g)}/{len(g.edges())} takes {pf(t1-t0, 3)}')
    return iter_array, t1 - t0

def bfs_successors(G, source, depth_limit=None):
    for p, c in nx.bfs_edges(G, source, depth_limit=depth_limit):
        yield (p, c)

def get_iter(g, source, print_ = False, nx_flag = True):
    # get iter_array efficiently
    t0 = time()
    if nx_flag:
        iter = list(nx.bfs_successors(g, source))
        iter_list = [[v] + nbrs for v, nbrs in iter]
        iter_array = np.array(list(itertools.zip_longest(*iter_list, fillvalue=0))).T

        iter_array = np.concatenate([iter_array, np.array(list(map(len, iter_list))).reshape(-1,1)], axis=1) # every row is of form [512 474   0 ...   0   0   2]
        iter_array = iter_array.astype(int)
    else:
        iter_array = np.array(list((bfs_successors(g, source))))

    t1 = time()
    if print_:
        print(iter_array)
        print(f'get_iter for graph of size {len(g)}/{len(g.edges())} takes {pf(t1-t0, 3)}')
    return iter_array, t1 - t0

def get_iter_fast(g, source):
    iter_array = np.array(list((bfs_successors(g, source))))
    return iter_array, 0



if __name__ == '__main__':
    np.random.seed(42)
    n =  2000
    n_iter = n
    G = nx.random_tree(n, seed=42) # nx.random_geometric_graph(100, 0.1)
    assert nx.is_connected(G)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.random()

    # t = 0
    # for source in range(n):
    #     _, t_ = get_iter(G, source, nx_flag=False, print_=False)
    #     t += t_
    # print(f'total get_iter takes {t}')
    # sys.exit()

    attr_dict = nx.get_edge_attributes(G, 'weight')
    attr_arr = np.zeros((n, n))
    for k, v in attr_dict.items():
        attr_arr[k[0]][k[1]] = v

    t1 = time()
    for source in range(n_iter):
        ultra_dict = bfs_fast_python(G, source)
        ultra_array_ = [0] * len(ultra_dict)
        for k, v in ultra_dict.items():
            ultra_array_[k] = v

        iter_array2, _ = get_iter_fast(G, source)
        ultra_array2 = bfs_faster2(attr_arr, iter_array2, n)
        assert (ultra_array_ == ultra_array2).all()

    t2 = time()
    print('*'*150)
    sys.exit()


    t1_, t2_ = 0, 0
    for source in range(n_iter):
        # iter_array1, _ = get_iter(G, source, print_ = False, nx_flag = True)
        # iter_array2, _ = get_iter(G, source, print_ = False, nx_flag = False)

        _t1 = time()
        iter_array2, _ = get_iter_fast(G, source)
        _t2 = time()

        # ultra_array1 = bfs_faster(attr_arr, iter_array1, n) #todo: array1 has some bugs
        ultra_array2 = bfs_faster2(attr_arr, iter_array2, n)

        # print(ultra_array2)
        # try:
        #     assert (ultra_array1 == ultra_array2).all()
        # except:
        #     print(ultra_array1)
        #     print(ultra_array2)
        #     print('-'*50)

        _t3 = time()

        t1_ += (_t2 -_t1)
        t2_ += (_t3 -_t2)

    sys.exit(f'python/bfs_faster takes {(pf(t2 - t1, 3))}/{(pf(t1_, 3))}+{(pf(t2_, 3))}')


    t0 = time()
    for source in range(n_iter):
        bfs_fast_complied(G, source)

    t1 = time()
    for source in range(n_iter):
        bfs_fast_python(G, source)
    t2 = time()

    print(list(nx.bfs_successors(G, 1))[:10])

    for source in range(n_iter):
        iter = nx.bfs_successors(G, source)
        iter = list(iter)
        bfs_fast_complied2(G, source, iter)
    t3 = time()



    attr_dict = nx.get_edge_attributes(G, 'weight')
    for source in range(n_iter):
        iter = nx.bfs_successors(G, source)
        iter = list(iter)
        bfs_fast_complied3(attr_dict, source, iter)
    t4 = time()

    attr_dict = nx.get_edge_attributes(G, 'weight')
    for source in range(n_iter):
        iter = nx.bfs_successors(G, source)
        iter = list(iter)
        bfs_fast_complied4(attr_dict, iter, n)
    t5 = time()

    # prepare
    attr_dict = nx.get_edge_attributes(G, 'weight')
    attr_arr = np.zeros((n, n))
    for k, v in attr_dict.items():
        attr_arr[k[0]][k[1]] = v
    t5 = time()
    for source in range(n_iter):
        iter = nx.bfs_successors(G, source)
        iter = list(iter)
        bfs_fast_complied5(attr_arr, iter, n)

    t6 = time()
    bfs_fast_complied6(G)
    t7 = time()


    print(f'python/complied/complied2/complied3/4/5 takes {(pf(t2-t1, 3))}/{pf(t1-t0, 3)}/{pf(t3-t2, 3)}/'
          f'{pf(t4-t3, 3)}/{pf(t5-t4, 3)}/{pf(t6-t5, 3)}/{pf(t7-t6, 3)}')
    sys.exit()

    assert primes_python_(100) == primes_python_complied(100)
    t0 = time()
    for _ in range(100):
        primes_python_complied(1000)

    t1 = time()
    for _ in range(100):
        primes_python_(1000)
        # primes(1000)
    t2 = time()

    print(f'python/complied takes {(t2-t1)}/{(t1-t0)}')
