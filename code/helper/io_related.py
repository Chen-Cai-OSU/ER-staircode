import time
import os
import pickle
import json

def make_dir(dir):
    # has side effect
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_obj(obj, dir, file):
    t0 = time.time()
    if not os.path.exists(dir): make_dir(dir)
    with open(dir + file, 'wb') as f:
        pickle.dump(obj, f)
        print('Saved graphs. Takes %s' % (time.time() - t0))

def load_obj(dir, file):
    t0 = time.time()
    with open(dir + file, 'rb') as f:
        pickle.load(f)
        print('Load file %s. Takes %s' % (str(file), time.time() - t0))

class io():
    def __init__(self, dir, file, saver = 'json'):
        self.dir = dir
        self.file = file
        self.obj = None
        self.saver = saver

    def save_obj(self, obj):
        self.obj = obj
        t0 = time.time()
        if not os.path.exists(self.dir): make_dir(self.dir)

        if self.saver=='pickle':
            with open(self.dir + self.file, 'wb') as f:
                pickle.dump(self.obj, f)
        elif self.saver == 'json':
            with open(self.dir + self.file, 'wb') as f:
                json.dump(self.obj, f)
        else:
            print('Saver is not specified. ')
            raise IOError
        print('Saved obj at %s. Takes %s' % (self.dir + self.file, time.time() - t0))

    def load_obj(self):
        t0 = time.time()
        try:
            if self.saver=='pickle':
                with open(self.dir + self.file, 'rb') as f:
                    res = pickle.load(f)
            elif self.saver == 'json':
                with open(self.dir + self.file, 'r') as f:
                    res = json.load(f)
            else:
                print('Saver is not specified. ')
                raise IOError

            print('Load file %s. Takes %s' % (str(self.file), time.time() - t0))
            return res
        except (IOError, EOFError) as e:
            print('file %s does not exist when loading with %s' % (self.dir + self.file, self.saver))
            return 0

    def rm_obj(self):
        if os.path.exists(self.dir + self.file):
            os.remove(self.dir + self.file)
            print('Delete file %s with success' % self.file)
        else:
            print('file %s does not exist' % self.file)

if __name__ == '__main__':
    import numpy as np
    from helper.intersection import staircase
    input = {0: {'type1': np.array([[1.80421575, 0.07641152]]), 'type2': None},
     1: {'type1': np.array([[1.8892676, 0.09383035], [1,2]]), 'type2': None},
     2: {'type1': np.array([[2.14402994, 0.01665461]]), 'type2': None},
     3: {'type1': np.array([[2.59201959, 0.08189007]]), 'type2': None}
     }

    different_juncs = []
    for k, v in input.items():
        single_juncs = v['type1'].tolist()
        different_juncs.append(single_juncs)

    for single_juncs in different_juncs:
        print(single_juncs)
        s = staircase()
        s.build_segs_from_juncs(single_juncs)
        s.find_intersect((1.01, 2))
        s.plot_segs()

