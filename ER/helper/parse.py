""" parse rivet's file """

import sys
import numpy as np
from pprint import pprint
dir = '/home/cai.507/Documents/DeepLearning/Clustering/'
file = f'{dir}circle_300pts_density.txt'

def parse_rivet(file):
    increase = True
    with open(file, 'r') as f:
        cont = f.readlines()

    for i, line in enumerate(cont):
        if line.startswith('#'): continue
        if line.startswith('points'):
            i += 1
            break

    d = int(cont[i])
    max_len_edge = float(cont[i + 1])
    ind = cont[i + 2]

    if ind.startswith('[-]'):
        # raise Exception('Only support filtering by density in increasing value')
        increase = False

    coordinates = cont[i + 3:]
    new_coordinates = []
    for coor in coordinates:
        coor_ = coor.split('\t')
        coor_ = [float(k) for k in coor_]
        new_coordinates.append(coor_)
    new_coordinates = np.array(new_coordinates)
    pts, f = new_coordinates[:, :-1], new_coordinates[:, -1]
    assert pts.shape[1] == d, 'shape mismatch'

    if not increase:
        print('Flip function value for filtering by density in decreasing value')
        f = -f

    print(f'shape of new coor is {new_coordinates.shape}/{pts.shape}/{f.shape}')
    return pts, f

if __name__ == '__main__':
    pts, f = parse_rivet(file)
    print(pts)
    print(f)