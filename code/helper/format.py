""" Format related function """

import numpy as np

def almostequal(x,y, threshold=1e-2):
    if abs(x-y) < threshold:
        return True
    else:
        return False

def print_line(n = 150):
    print('-'*n)

def precision_format(nbr, precision=1):
    # assert type(nbr)==float
    return  round(nbr * (10**precision))/(10**precision)

def rm_zerocol(data, cor_flag=False, print_flag=False):
    # data = np.zeros((2,10))
    # data[1,3] = data[1,5] = data[1,7] = 1
    n_col = np.shape(data)[1]
    del_col_idx = np.where(~data.any(axis=0))[0]
    remain_col_idx = set(range(n_col)) - set(del_col_idx)
    correspondence_dict = dict(zip(range(len(remain_col_idx)), remain_col_idx))
    inverse_correspondence_dict = dict(zip(remain_col_idx, range(len(remain_col_idx))))

    x = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)

    if print_flag:
        print('the shape before removing zero columns is %s' % (np.shape(data)))
        print('the shape after removing zero columns is %s'%(np.shape(x)))

    if cor_flag == True:
        return (x, correspondence_dict, inverse_correspondence_dict)
    else:
        return x

def normalize_(x, axis=0):
    from sklearn.preprocessing import normalize
    return normalize(x, axis=axis)

