import copy
import os
import pickle
from pathlib import Path
from joblib import Parallel, delayed
import sys

from I_x.dgm2d import D_x_slice
from Esme.helper.format import precision_format as pf

from I_x.dgm2d import get_idxsort, set_graph, get_subgraph, update_subgraph, D_x_slice_clean, dgm_format
from gen.example import density
from gen.ptc_model import pts_on_square, toy_dataset
from gen.util import viz_stairs_
from viz import viz_pd

BACKEND = 'multiprocessing'

# todo: if noise is in test dir, there is some problem. If nosie.py is in I_x, everything is fine.

if __name__ == '__main__':
    show_flag = False
    save_flag = False
    verbose = 2

    # specify toy models
    # points, distm = pts_on_square(add_noise=True, n_noise=10)
    points, color, distm = toy_dataset(n_sample=100, name='moon')
    if show_flag: viz_pd(points, show=True, color=None)

    f = - density(points).reshape(len(points), 1)
    f_sort = copy.deepcopy(f).reshape((len(f),)).tolist() # # f_sort is a global variable
    f_sort.sort()
    idx_sort = get_idxsort(f, f_sort)
    sigmas = f.reshape(len(f),).tolist()
    sigmas.sort()

    G = set_graph(f.shape[0], f, distm)
    G = get_subgraph(f, 0, distm)

    for i in range(len(sigmas)-1): # sigmas = np.linspace(0, 1, 10)
        G  = update_subgraph(G, f, distm, sigmas[i], sigmas[i+1])

        home = str(Path.home())
        file = os.path.join(home, 'Documents', 'DeepLearning', 'Clustering', 'subgraphs', str(i+1) + '.pkl')
        with open(file, 'wb') as handle:
            print(f'Saving graph of {len(G.nodes())}/{len(G.edges())}')
            pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dgm = []
    for sigma in sigmas[1:]:
        dgm_slice = D_x_slice_clean(f, f_sort, idx_sort, sigma, sigmas) # [{key: (sigma, epsilon)}, ...] # todo: only works for D_x_slice_clean, not D_x_slice
        dgm.append(dgm_slice) # dgm is a list of dict of form {x:(sig,eps), ...}

    stairs0 = Parallel(n_jobs=-1, backend=BACKEND)(delayed(D_x_slice)(f, f_sort, idx_sort, sigma, sigmas, mst_opt=False, print_=False) for sigma in sigmas[1:])

    res = dgm_format(stairs0, filter = True)
    # sys.exit()

    # for i in range(len(dgm)):
    #     try:
    #         assert stairs0[i] == dgm[i]
    #     except:
    #         print(f'{i} is different. \n stairs0{i} is {stairs0[i]}. \n dgm{i} is {dgm[i]}')
    assert stairs0 == dgm
    viz_stairs_(res, alpha=.05, color = color, yscale='linear')
    # viz_stairs_(res, alpha=.05, color=color, yscale='log')