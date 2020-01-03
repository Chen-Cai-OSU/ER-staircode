import matplotlib.pyplot as plt
import numpy as np
import sys
from ER.helper.io_related import make_dir
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def write_pts_for_viz(pts, color, name = 'moon'):
    # write points for interactive viz
    """
    :param pts: shape of (n, d)
    :return:
    """
    if pts.shape[1] > 2:
        print(f'interactive visualization works only for 2D data now...')
    x, y = pts[:, 0], pts[:, 1]
    color_set = list(set(color))
    color_set.sort()
    color2int = dict(zip(color_set, range(len(color_set))))
    color = [color2int[c] for c in color]
    print(f'Convert color to int according to {color2int}')
    dir = './viz/RuiLi/data'

    urls = []
    for i in range(pts.shape[0]):
        urls.append('public/image/' + name + '_' + str(i) + '.png')

    d = {'x': x, 'y': y, 'Label': color, 'url': urls}
    df = pd.DataFrame(data=d)

    csv_file = get_name_for_csv(name=name)
    df.to_csv(csv_file, index=False)
    print(f'Save to {csv_file}...')

def get_name_for_csv(name='moon'):
    if name == 'moon':
        csv_file = './viz/RuiLi/data/tsne.csv'
    elif name == 'blob':
        csv_file = './viz/RuiLi/data/pca.csv'
    elif name == 'circle':
        csv_file = './viz/RuiLi/data/mds.csv'
    else:
        raise Exception(f'No such {name}')
    return csv_file

def get_left_epsilon(stair, sigma = -1):
    """
    :param stair: is a list of tupe of form (sigma, epsilon)
    :return: from all left(smaller) epsion, get the the rightmost(largest)
    """
    assert type(stair) == list

    if len(stair) == 0:
        return None, -1
    else: # todo can be improved
        eps, sigs = [], []
        for (sig, ep) in stair:
            if sig < sigma:
                eps.append(ep)
                sigs.append(sig)
        if len(eps) == 0: # all points are on the right
            return None, -1
        else:
            return sigs[eps.index(min(eps))], min(eps)

def get_epsilon(stair, sigma):
    """
    #todo: make it faster
    for a stair, find the largest sig smaller than, and return its corresponding epsilon
    :param stair: a list of tuples
    :return:
    """
    stair.sort(key=lambda x:x[0])
    n = len(stair)
    epsilon = -1e5
    for i in range(n):
        if stair[i][0] < sigma + 1e-6:
            epsilon = stair[i][1]

    return epsilon

def get_previous_epsilon(stair):
    assert type(stair) == list

    if len(stair) == 0:
        return None
    else:
        return stair[-1][1]

def viz_stair(stair, show=False, title=None, save = False, **kwargs):
    """
    :param stair: a list of tuple of form (sigma, epsilon)
    :param plot: if True, convert to new_stair and visualize
    :return: if plot False, return new stair
    """
    if stair == []: return [(0,0)]
    for i in range(1, len(stair)):
        assert stair[i][0] >= stair[i-1][0]
        assert stair[i][1] <= stair[i][1]
    new_stair = []

    new_stair.append((stair[0][0], 0)) # the fist point is  (f(x), 0)
    new_stair.append(stair[0])
    for i in range( len(stair)):
        new_stair.append(stair[i])
        if i+1 < len(stair) and stair[i+1][1] < stair[i][1]:
            new_pt = (stair[i+1][0], stair[i][1])
            new_stair.append(new_pt)

    new_stair = np.array(new_stair)
    plt.plot(new_stair[:, 0], new_stair[:, 1], 'b-')
    if title is not None: plt.title(title)

    if save:
        make_dir(kwargs['dir'])
        file = os.path.join(kwargs['dir'], kwargs['f'])
        print(f'save at {file}')
        plt.savefig(file)
    if show: plt.show()

    else:
        return new_stair

def viz_stairs(stairs):
    # stairs = []
    # stair1 = [(1, 2), (3, 1.7), (4, 1.5), (10, 0.9)]
    # stair2 = [(1, 3.1), (3, 2.3), (4.7, 1), (11, 0.8)]
    # stairs.append(stair1)
    # stairs.append(stair2)

    for i in range(len(stairs)):
        stairs[i] = np.array(viz_stair(stairs[i], plot=False))

    for i in range(len(stairs)):
        plt.plot(stairs[i][:,0], stairs[i][:,1], 'b-')

    plt.show()

def density(data):
    """
    :param data: np.array of shape (n, d)
    :return: density for each point
    """
    from sklearn.neighbors.kde import KernelDensity
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    return kde.score_samples(data)

def expand_list(l):
    """
    :param l: a list of lists
    :return: a expanded list
    """
    res = []
    for lis in l:
        res += lis
    return res

def I_x(f, distm, x, sigma):
    """
    :param f: array of shape (n, 1)
    :param distm: dist matrix of shape (n, n)
    :param x: idx
    :param sigma: sigma considered for now
    :return:
    """
    n = f.shape[0]
    indices_below = [i for i in range(n) if f[i][0] < sigma]
    assert f[x] <= sigma

    import networkx as nx
    tor = 1e-6
    filter_m = np.outer([f < sigma + tor], [f < sigma + tor])  # todo: viz filter_m should be square
    val = np.multiply(distm, filter_m)
    edgelist = np.argwhere(val > 0)  # np.array of shape (_, 2)
    val = val[val > 0]

    n_ = edgelist.shape[0]  # num of edges in sub-complete graph
    assert edgelist.shape[0] == val.shape[0]
    edgelist = np.concatenate((edgelist, val.reshape((n_, 1))), axis=1)  # array of shape (_, 3)
    assert edgelist.shape[1] == 3

    # format data for nx
    lines = []
    for i in range(n_):
        s, t, weight = edgelist[i][0], edgelist[i][1], edgelist[i][2]
        line = f"{int(s)} {int(t)}" + ' {' + f"'weight':{float(weight)}" + '}'
        lines.append(line)

    G = nx.parse_edgelist(lines, nodetype=int)

    mst = nx.minimum_spanning_tree(G)
    path_dict = nx.single_source_shortest_path(mst, x)  # [1, 70, 78, 13, 10]
    path_dict = dict((k, path_dict[k]) for k in indices_below if k in path_dict)
    length_dict = {}

    # todo try to use single_path_length
    for k, v in path_dict.items(): # v is like [10, 7, 0]
        if len(v) > 1:
            lenlist = [mst[v[i]][v[i+1]]['weight'] for i in range(len(v)-1)]
        else:
            lenlist = [1e10]
        length_dict[k] = lenlist

    epsion = min(expand_list(list(length_dict.values())))

    return sigma, epsion

def viz_stair_(stair, show = True, print_=False, flip_y = False, hide_tickers = False,
               text = False, num = True, save = False, **kwargs):
    """
    :param stair: a list of tuples [(1, 2), (3, 1.7), (4, 1.5), (10, 0.9)]
    :param show: flag
    :param print_: flag
    :param kwargs: color, alpha...
    :return:
    """
    if stair == []: return
    x = np.array([p[0] for p in stair]) # np.linspace(0, 5, 10)
    y = np.array([p[1] for p in stair])
    if flip_y: y = -y

    if print_:
        print_(f'First five x {x[:5]}')
        print_(f'First five y {y[:5]}')
    color = kwargs.get('color', 'blue')
    alpha = kwargs.get('alpha', 0.4)
    title = kwargs.get('title', None)
    yscale = kwargs.get('yscale', 'linear')

    juncx, juncy, bi1 = bigrad1(stair, print_flag=False)
    bi2 = bigrad1(stair, bi2_flag=True, print_flag=False)
    # todo: color junc points
    # if bi2 is not None:
    #     plt.scatter(bi2[:, 0], bi2[:,1], c='r')
    # plt.scatter(juncx, juncy, c='b')

    plt.fill_between(x, y, step="post", alpha=alpha, color=color, edgecolor=color)
    plt.yscale(yscale)
    if text:
        pass
        # left, width = .25, .5
        # bottom, height = .25, .5
        # right = left + width
        # top = bottom + height
        # plt.text(0.5 * (left + right), 0.5 * (bottom + top), 'middle',
        #         horizontalalignment='center',
        #         verticalalignment='center',
        #         fontsize=2, color='red',
        #         )
    if title is not None: plt.title(title, position=(0.3, 0.3))

    if hide_tickers:
        plt.xticks([])
        plt.yticks([])

    if save:
        data_name = str(kwargs.get('name', 'None'))
        dir = './viz/RuiLi/public/image/' + data_name + '_'
        make_dir(dir)
        name = str(kwargs.get('ind_title', '1.png'))
        plt.title(f'{data_name}: idx {name}')
        plt.savefig(dir + name)
        print(f' Save at {dir + name}')

    if show: plt.show()
    # plt.close() # todo: may need to turn on
    junc_pts = {'type1': bi1, 'type2': bi2}
    return junc_pts


def viz_stairs_(stairs, title=None, flip = False, choices = None, save = False, hide_tickers = True, **kwargs):
    """
    :param stairs: a list of list of tuples
    :param title:
    :param kwargs: alpha: either a number or a list of num
                   color: either a color or a list of colors
    :return:
    """
    default_colors = ['blue'] * len(stairs)
    default_alphas = [0.05] * len(stairs)
    default_yscale = 'linear'
    flip_color = None

    for i, stair in enumerate(stairs):
        try:
            color = kwargs.get('color', default_colors)[i]
        except:
            color = kwargs.get('color', default_colors)

        try:
            alpha = kwargs.get('alpha', default_alphas)[i]
        except:
            alpha = kwargs.get('alpha', default_alphas)

        yscale = kwargs.get('yscale', default_yscale)

        if not flip:
            flip_y = False
        else:
            if flip_color is None: flip_color = color
            flip_y = True if color == flip_color else False

        if choices == None:
            viz_stair_(stair, show=False, color=color, flip_y = flip_y, alpha=alpha, yscale=yscale, save=save, title=title)
            if title is not None: plt.title(title)

        elif i in choices: # only visualize a subset of stairs
            num=kwargs.get('num', False)
            title = str(i) if num == True else None
            junc = viz_stair_(stair, show=False, color=color, flip_y=flip_y, alpha=alpha, yscale=yscale,
                       hide_tickers=hide_tickers, save=save, ind_title=str(i), name=kwargs.get('name', 'None'))
            return junc
            # print(f'plot a single stair {i}')



def south_west_check(stair, x = 1, y = 1):
    # find all points that is south-west to point (x, y) in the stair
    n = len(stair)
    res = [stair[i] for i in range(n) if (stair[i][0] <= x and stair[i][1] <=y)]
    if len(res) > 1: # only contains junction point itself
        return False
    else:
        return True


def bigrad1(stair, print_flag = False, bi2_flag = False):
    junc = []
    for (sig, eps) in stair:
        if south_west_check(stair, x=sig, y=eps):
            junc.append((sig, eps))
    junc = np.array(junc)
    bi2 = bi12bi2(junc)
    if bi2_flag:
        return bi2

    junc_x, junc_y = junc[:, 0], junc[:, 1]
    if print_flag:
        print(f'There are {len(stair)} points in stair but {junc.shape[0]} points as Bigraded-1')
        print(junc)
        print(f'Now print out Bigraded-2')
        bi2 = bi12bi2(junc)
        print(bi2)
        print('-'*20)

    return junc_x, junc_y, junc

def bi12bi2(bi1):
    # bi1 is an array of size (n, 2)
    # [[2.2339768  0.16472838]
    #  [2.35852084 0.16436426]]
    n, d = bi1.shape
    if n < 2:
        return None
    else:
        res = []
        for i in range(n-1):
            new_coor = [bi1[i+1][0], bi1[i][1]]
            res.append(new_coor)
        return np.array(res)



if __name__ == '__main__':
    n = 50
    f = np.random.random((n, 1))
    f[10] = 0.1
    distm = np.random.random((n, n))
    distm = distm + distm.T
    x = 10
    sigma = 0.2
    sigmas = [f[i][0] for i in range(n) if f[i][0] > sigma]
    sigmas.sort()

    stair = []
    for sigma in sigmas:
        sig, eps = I_x(f, distm, x, sigma)
        stair.append((sig, eps))

    viz_stair_(stair)

    # viz_stair(stair, plot=True, show=True)

    sys.exit()

    viz_stairs(None)
    sys.exit()
    stair = [(1,2),(3,1.7), (4,1.5),(10,0.9)]
    stair_2 = [(1,3.1),(3,2.3), (4.7,1),(11,0.8)]

    stair = viz_stair(stair, plot=False)
    print(stair)
    # stair_2 = viz_stair(stair_2, plot=False)
    plt.plot(np.array(stair)[:,0], np.array(stair)[:,1], 'b-')
    # plt.plot(np.array(stair_2)[:, 0], np.array(stair_2)[:, 1], 'r-')

    plt.show()
    sys.exit()

    sig, epsilon = get_left_epsilon(stair, sigma=3.5)
    print(sig) # 3
    assert epsilon == 1.7

    sigma, epsilon = get_left_epsilon(stair, sigma=4.5)
    print(sigma) # 4
    assert epsilon == 1.5

    sigma, epsilon = get_left_epsilon(stair, sigma=10.5)
    print(sigma)# 10
    assert epsilon == 0.9

