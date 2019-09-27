from sklearn.neighbors.kde import KernelDensity
from ptc_model import two_cycle
from sklearn.neighbors.kde import KernelDensity
from Esme.helper.format import precision_format as pf

from ptc_model import two_cycle
from I_x.I_x_slice import I_x_slice_
from util import viz_stair, get_epsilon
from viz import viz_pd

import time
import sys
from joblib import Parallel, delayed
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def density(data):
    """
    :param data: np.array of shape (n, d)
    :return: density for each point
    """
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    return kde.score_samples(data)


def get_sigmas(f, idx):
    n = f.shape[0]
    sigma = f[idx][0]
    sigmas = [f[i][0] for i in range(n) if f[i][0] > sigma]
    sigmas.sort()
    return sigmas

def get_stair(f, distm, idx=1, method='heu', **kwargs):

    """
    :param f: array of shape (n, 1)
    :param distm: array of shape (n, 1)
    :param idx: point idx
    :param method: 'heu', 'bf','bf_f'
    :return: stair (a list of tuples)
    """

    sigmas = get_sigmas(f, idx)
    if method == 'bf':
        stair = []
        for sigma in sigmas:
            sig, eps = I_x_slice_(f, distm, idx, sigma, print_=False)
            stair.append((sig, eps))

    elif method == 'bf_p':
        stair = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(I_x_slice_)(f, distm, idx, sigma, print_=False) for sigma in sigmas)

    elif method == 'heu':
        stair = []
        k = 0
        n_sigmas = len(sigmas)
        while k < n_sigmas-1:
            dist = n_sigmas-1 - k
            assert dist > 0
            jump_range = [dist//2, dist//4, dist//8, min(4, dist), 1] # todo tunning
            jump_range = list(set(jump_range))
            jump_range = [i for i in jump_range if i!=0]
            jump_range.sort(reverse=True)

            stair.sort(key=lambda x: x[0])
            if kwargs.get('p_flag', False): print(f'k is now {k}. n_sigmas is {n_sigmas}')
            for j in jump_range:
                sig, eps = I_x_slice_(f, distm, idx, sigmas[k + j], print_=False)
                if len(stair) >= 1 and eps == get_epsilon(stair, sigmas[k]): # stair[-1][1]
                    stair.append((sig, eps))
                    if kwargs.get('p_flag', False): print(f'in the break loop {k+j}')
                    break
                else:
                    stair.append((sig, eps))
                    # stair.sort(key = lambda x: x[0])
            k += j

    else:
        raise Exception(f'No such method {method}')

    return stair


parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=6, type=int, help='point index') # (1515, 500) (3026,)
parser.add_argument("--n", default=50, type=int, help='num of points') # (1515, 500) (3026,)
parser.add_argument("--method", default='heu', type=str, help='bf (brute force), heu (heuristic)') # (1515, 500) (3026,)

if __name__ == '__main__':
    start_t = time.time()
    args = parser.parse_args()
    idx = args.idx
    show_flag = True
    save_flag = False

    # specify toy models
    points, distm = two_cycle(args.n)
    if show_flag: viz_pd(points, show=True, color=None, annotate=[10, 100])

    f = - density(points).reshape(len(points),1)

    stair = get_stair(f, distm, idx=idx, method='heu')

    # viz
    end_t = time.time()
    title = f'I_x for {idx} with co-density {pf(f[idx][0], 2)}. Comp.T: {pf(end_t - start_t, 1)}'
    viz_stair(stair, show=show_flag, title=title, save=save_flag, dir='./img/', f=f'{idx}.png')

    sys.exit()



    sigma = f[idx][0]
    sigmas = [f[i][0] for i in range(n) if f[i][0] > sigma]
    sigmas.sort()

    stair = []

    if args.method == 'heu':
        k = 0
        n_sigmas = len(sigmas)
        while k < n_sigmas-1:
            dist = n_sigmas-1 - k
            assert dist > 0
            jump_range = [dist//2, dist//4, dist//8, min(4, dist), 1] # todo tunning
            jump_range = list(set(jump_range))
            jump_range = [i for i in jump_range if i!=0]
            jump_range.sort(reverse=True)
            print(jump_range)

            stair.sort(key=lambda x: x[0])
            print(f'k is now {k}. n_sigmas is {n_sigmas}')
            for j in jump_range:
                sig, eps = I_x_slice_(f, distm, idx, sigmas[k + j], print_=False)
                if len(stair) >= 1 and eps == get_epsilon(stair, sigmas[k]): # stair[-1][1]
                    stair.append((sig, eps))
                    print(f'in the break loop {k+j}')
                    break
                else:
                    stair.append((sig, eps))
                    # stair.sort(key = lambda x: x[0])
            k += j

    elif args.method == 'bf':
        for sigma in sigmas:
            sig, eps = I_x_slice_(f, distm, idx, sigma, print_=False)
            stair.append((sig, eps))

    elif args.method == 'bf_':
        n_sigmas = len(sigmas)
        stair = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(I_x_slice_)(f, distm, idx, sigma, print_=False) for sigma in sigmas)

