import matplotlib.pyplot as plt
import numpy as np
import sys
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

def get_previous_epsilon(stair):
    assert type(stair) == list

    if len(stair) == 0:
        return None
    else:
        return stair[-1][1]

def viz_stair(stair, plot=False):
    """
    :param stair: a list of tuple of form (sigma, epsilon)
    :param plot: if True, convert to new_stair and visualize
    :return: if plot False, return new stair
    """

    for i in range(1, len(stair)):
        assert stair[i][0] >= stair[i-1][0]
        assert stair[i][1] <= stair[i][1]
    new_stair = []

    new_stair.append(stair[0])
    for i in range( len(stair)):
        new_stair.append(stair[i])
        if i+1 < len(stair) and stair[i+1][1] < stair[i][1]:
            new_pt = (stair[i+1][0], stair[i][1])
            new_stair.append(new_pt)

    if plot == True:
        new_stair = np.array(new_stair)
        plt.plot(new_stair[:,0], new_stair[:,1], 'b-')
        plt.show()
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


if __name__ == '__main__':
    viz_stairs()
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

