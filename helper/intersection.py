import sys
from helper.format import precision_format as pf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc
from functools import cmp_to_key
np.random.seed(42)

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)
    plt.show()

class staircase():
    def __init__(self):
        self.segs = []
        self.exd = 10 # extend the rightmost point. Ideally should be infinity
        self.line = None

    def addseg(self, seg):
        """
        :param seg: ((x1, y1),(x2, y2))
        """
        self._check_direction(seg)
        self.segs.append(seg)


    def build_segs_from_juncs(self, juncs, verbose=0):
        """
        :param juncs: a list of tuples (junction points)
        :return: build segs
        """

        juncs.sort(key=lambda x: x[0])
        self.addseg(((juncs[0][0], 0),juncs[0]))
        if verbose>0: print(f'juncs is {juncs}')
        filter_juncs = [juncs[0]]

        for p in juncs[1:]:
            if p[1] == filter_juncs[-1][1]: # if new point has the same y with previous point in filter_juncs
                pass # more for debug
            elif p[0] == filter_juncs[-1][0]: # if new point has the same y with previous point in filter_juncs
                pass
            else:
                pt1 = (p[0], filter_juncs[-1][1])
                filter_juncs.append(pt1)
                filter_juncs.append(p)

        last_pt = (filter_juncs[-1][0] + self.exd, filter_juncs[-1][1])
        filter_juncs.append(last_pt)
        if verbose>0: print(filter_juncs)
        n = len(filter_juncs)
        assert n>1, f'Cardinality of filtered Juncs is {n}'
        for i in range(n-1):
            p, q = filter_juncs[i], filter_juncs[i+1]
            seg = (p, q)
            seg = self._check_seg_rep(seg)
            self.addseg(seg)


    def _check_direction(self, seg):
        p1, p2 = seg
        if p1[1] == p2[1]:
            return 'hor'
        elif p1[0] == p2[0]:
            return 'ver'
        else:
            sys.exit(f'Seg {seg} has to be either horizonal or vertical.')

    def _check_right_bottom(self, p1, p2):
        # check if p2 is right bottom to p1
        p1_x ,p1_y = p1
        p2_x, p2_y = p2
        if p2_x>= p1_x and p2_y <= p1_y:
            return True
        else:
            return False

    def _rep_line(self, l):
        a, b = l
        return f'y={a}*x+{b}'

    def _rep_pt(self, pt):
        return f'({pf(pt[0], 2)}, {pf(pt[1], 2)})'

    def _rep_seg(self, seg):
        p1, p2 = seg
        return self._rep_pt(p1) + self._rep_pt(p2)

    def _check_seg_rep(self, seg):
        """
        reverse the order of two ends of seg, reverse if there is error
        :param seg:
        :return:
        """
        self._check_direction(seg)
        p1, p2 = seg
        if p2[0] >= p1[0]:
            seg = (p2, p1)

        p1, p2 = seg
        if p2[1] <= p1[1]:
            seg = (p2, p1)
        return seg

    def _cmp_seg(self, s1, s2):
        p1 = s1[1] if self._check_direction(s1) == 'hor' else s1[0]
        p2 = s2[1] if self._check_direction(s2) == 'hor' else s2[0]
        return self._check_right_bottom(p1, p2)

    def sort_segs(self):
        sorted_segs = sorted(self.segs, key=cmp_to_key(self._cmp_seg))
        print(f'original segs is {self.segs}')
        print(f'sorted segs is {sorted_segs}')


    def _check_intersect(self, l, seg, verbose = 0):
        """
        check if line l intersects with line segments
        :param l: a tupe of (a, b) representing line y=ax+b
        :param seg: (p1, p2) where both p1 and p2 are 2d tuple where p2[x]>=p1[x]
        :return: ture or false
        """
        p1, p2 = seg
        assert p2[0] >= p1[0], 'Order of two ends of segment needs to be reversed.'
        assert p2[1] >= p1[1], 'Order of two ends of segment needs to be reversed.'
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        a, b = l[0], l[1]
        if self._check_direction(seg) == 'hor':
            if a * x1 + b <= y1 and a * x2 + b >= y2:
                intersect_pt = ((y1-b)/float(a), y1)
                if verbose > 0: print(f'line {self._rep_line(l)} instersects with horizonal seg {self._rep_seg(seg)} at {self._rep_pt(intersect_pt)}')
                return True
            else:
                if a * x1 + b > y1 and a * x2 + b > y2:
                    return 'high' # line is above seg
                elif a * x1 + b < y1 and a * x2 + b < y2:
                    return 'low' # line is above seg
                else:
                    sys.exit('Unconsidered Case')

        else: # 'ver'
            if a * x1 + b >= y1 and a * x2 + b <= y2:
                intersect_pt = (x1, a*x1+b)
                if verbose > 0: print(f'line {self._rep_line(l)} instersects with vertical seg {self._rep_seg(seg)} at {self._rep_pt(intersect_pt)}')
                return True
            else:
                if a * x1 + b > y1 and a * x2 + b > y2:
                    return 'high'  # line is above seg
                elif a * x1 + b < y1 and a * x2 + b < y2:
                    return 'low'  # line is above seg
                else:
                    sys.exit('Unconsidered Case')

    def addline(self, l):
        a, b = l
        def my_formula(x):
            return a * x + b
        self.line = my_formula

    def _check_segs_ordered(self):
        pass

    def find_instersect(self, l):
        self.addline(l)

        for seg in self.segs: # todo: implement binary search here
            self._check_intersect(l, seg, verbose=1)



    def plot_segs(self):
        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        new_segs = []
        for seg in self.segs:
            new_seg = [seg[0], seg[1]]
            new_segs.append(new_seg)
            # segs = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]

        lc = mc.LineCollection(new_segs, linewidths=1)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        x_min, x_max = ax.get_xlim()
        if self.line is not None:
            graph(self.line, range(int(x_min), int(x_max)))

        plt.show()

def gen_juncs(n=10, seed=42):
    """ generate junction(type-1) points """
    np.random.seed(seed)
    p = (0, 20)
    juncs = [p]
    for i in range(n-1):
        incx, decy = np.random.random(), np.random.random()
        p = (p[0]+incx, p[1]-decy)
        juncs.append(p)
    return juncs


import fire

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--line_a", default=1.01, type=float, help='y=ax+b')
parser.add_argument("--line_b", default=1, type=float, help='y=ax+b')

if __name__ == '__main__':
    args = parser.parse_args()
    n = args.n

    for i in range(2):
        print(i)
        s = staircase()
        juncs = gen_juncs(100, seed=i)
        juncs = [(0, 10),(1,9),(3,5), (2,8)]
        s.build_segs_from_juncs(juncs)
        # s.sort_segs()
        s.find_instersect((args.line_a, args.line_b))
        s.plot_segs()

