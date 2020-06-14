import sys
from helper.format import precision_format as pf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc
from functools import cmp_to_key
from time import time

np.random.seed(42)
EXT = 3

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)
    plt.show()

class staircase():
    def __init__(self):
        self.segs = []
        self.aug_segs = []
        self.exd = EXT # extend the rightmost point. Ideally should be infinity
        self.line = None
        self.null_data = {'killer': 'null'}

    def addseg(self, seg):
        """
        :param seg: ((x1, y1),(x2, y2))
        """
        self._check_direction(seg)
        self.segs.append(seg)

    def add_augseg(self, augseg):
        """ add augmented segment """
        seg = augseg[:2]
        self._check_direction(seg)
        self.aug_segs.append(augseg)

    def build_segs_from_juncs(self, juncs, verbose=0):
        """
        :param juncs: a list of tuples (junction points)
        :return: build segs
        """

        juncs.sort(key=lambda x: x[0])
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

        last_pt = (max(filter_juncs[-1][0] + self.exd, 10), filter_juncs[-1][1]) # small hack to make the gui looks better
        filter_juncs.append(last_pt)
        if verbose>0: print(filter_juncs)
        n = len(filter_juncs)
        assert n > 1, f'Cardinality of filtered Juncs is {n}'
        for i in range(n-1):
            p, q = filter_juncs[i], filter_juncs[i+1]
            seg = (p, q)
            aug_seg = (p, q, {'killer': np.random.randint(n)})

            seg = self._check_seg_rep(seg)
            self.addseg(seg)

            aug_seg = self._check_seg_rep(aug_seg, aug=True)
            self.add_augseg(aug_seg)

        left_most_seg = ((juncs[0][0], 0),juncs[0])
        bottom_seg = ((juncs[0][0], 0), (10, 0)) # bottom_seg = ((juncs[0][0], 0), ) # small hack to make the gui looks better
        self.default_segs = [left_most_seg, bottom_seg] # check this two segs seprately

        # self.addseg(left_most_seg) # add leftmost seg
        # self.addseg(bottom_seg)
        # self.add_augseg(left_most_seg + (self.null_data,))
        # self.add_augseg(bottom_seg + (self.null_data,))

    def _check_direction(self, seg):
        """ check seg to be horizonal or vertical """
        p1, p2 = seg[:2]
        if p1[1] == p2[1]:
            return 'hor'
        elif p1[0] == p2[0]:
            return 'ver'
        else:
            sys.exit(f'Seg {seg} has to be either horizonal or vertical.')

    def _check_right_bottom(self, p1, p2):
        """ check if p2 is right bottom to p1 """
        p1_x ,p1_y = p1
        p2_x, p2_y = p2
        if p2_x>= p1_x and p2_y <= p1_y:
            return True
        else:
            return False

    def _rep_line(self, l):
        a, b = l
        return 'y={:2f}*x+{:2f}'.format(a, b)

    def _rep_pt(self, pt):
        return f'({pf(pt[0], 2)}, {pf(pt[1], 2)})'

    def _rep_seg(self, seg):
        p1, p2 = seg[:2]
        return self._rep_pt(p1) + self._rep_pt(p2)

    def _check_seg_rep(self, seg, aug=False):
        """
        reverse the order of two ends of seg, reverse if there is error
        :param seg:
        :return:
        """
        self._check_direction(seg)
        p1, p2 = seg[:2]
        if aug: data = seg[2]
        if p2[0] >= p1[0]:
            seg = (p2, p1) if not aug else (p2, p1, data)

        p1, p2 = seg[:2]
        if p2[1] <= p1[1]:
            seg = (p2, p1) if not aug else (p2, p1, data)
        return seg

    def _cmp_seg(self, s1, s2):
        p1 = s1[1] if self._check_direction(s1) == 'hor' else s1[0]
        p2 = s2[1] if self._check_direction(s2) == 'hor' else s2[0]
        return self._check_right_bottom(p1, p2)

    def sort_segs(self):
        sorted_segs = sorted(self.segs, key=cmp_to_key(self._cmp_seg))
        print(f'original segs is {self.segs}')
        print(f'sorted segs is {sorted_segs}')

    def check_augseg(self, augseg):
        """ check if a seg is augmented or not """
        if len(augseg) == 3 and type(augseg[2]) is dict:
            return True
        else:
            return False

    def _check_intersect(self, l, seg, verbose = 0):
        """
        check if line l intersects with line segments
        :param l: a tupe of (a, b) representing line y=ax+b
        :param seg: (p1, p2) where both p1 and p2 are 2d tuple where p2[x]>=p1[x]
        :return: ture or false
        """

        if self.check_augseg(seg):
            p1, p2, data = seg
        else:
            p1, p2 = seg
            data = 'null'

        assert p2[0] >= p1[0], f'Order of two ends of segment {self._rep_seg(seg)} needs to be reversed.'
        assert p2[1] >= p1[1], f'Order of two ends of segment {self._rep_seg(seg)} needs to be reversed.'
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        a, b = l[0], l[1]
        if self._check_direction(seg) == 'hor':
            if a * x1 + b <= y1 and a * x2 + b >= y2:
                intersect_pt = ((y1-b)/float(a), y1)
                if verbose > 0: print(f'line {self._rep_line(l)} instersects with horizonal seg {self._rep_seg(seg)} at {self._rep_pt(intersect_pt)}.')
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
                if verbose > 0: print(f'line {self._rep_line(l)} instersects with vertical seg {self._rep_seg(seg)} at {self._rep_pt(intersect_pt)}.')
                return True
            else:
                if a * x1 + b > y1 and a * x2 + b > y2:
                    return 'high'  # line is above seg
                elif a * x1 + b < y1 and a * x2 + b < y2:
                    return 'low'  # line is above seg
                else:
                    sys.exit('Unconsidered Case')

    def setline(self, l):
        """ set a line. for plotting """
        a, b = l
        def my_formula(x):
            return a * x + b
        self.line = my_formula

    def _check_twosegs(self, seg1, seg2):
        """ assert seg2 is larger(right bottom) to seg1 """
        seg1, seg2 = seg1[:2], seg2[:2]
        seg1_rightmost = max(seg1[0][0], seg1[1][0])
        seg1_bottommost = min(seg1[0][1], seg1[1][1])
        seg2_rightmost = max(seg2[0][0], seg2[1][0])
        seg2_bottommost = min(seg2[0][1], seg2[1][1])
        assert seg2_rightmost >= seg1_rightmost, f'seg2 {self._rep_seg(seg2)} not at right to seg1 {self._rep_seg(seg1)} '
        assert seg2_bottommost <= seg1_bottommost, f'seg2 {self._rep_seg(seg2)} not at bottom to seg1 {self._rep_seg(seg1)} '

    def _check_segs_ordered(self, segs):
        """ check segs are ordered (staircase from left to right, top to bottom)"""
        n = len(segs)
        for i in range(n-1):
            seg1, seg2 = segs[i], segs[i+1]
            self._check_twosegs(seg1, seg2)
        # print('pass segs order test')

    def _find_default_intersect(self, l):
        """ test intersection with two default segs with line l """
        segs = self.default_segs # haven't added aug version
        for seg in segs:
            self._check_intersect(l, seg, verbose=1)

    def find_intersect(self, l, aug=False):
        """ brute force search """
        self._find_default_intersect(l)
        self.setline(l)
        segs = self.aug_segs if aug else self.segs

        t0 = time()
        for seg in segs:
            if self._check_intersect(l, seg, verbose=1) == True:
                print(f'Intersect at seg index {segs.index(seg)} {self._rep_seg(seg)}')
        # print(f'line search takes {time()-t0}')

    def find_intersect_binary(self, l, aug = False, verbose = 0, check = True):
        """ binary search """
        self._find_default_intersect(l)
        self.setline(l)
        segs = self.aug_segs if aug else self.segs

        # todo implelemt binary. Done.
        if check: self._check_segs_ordered(segs)
        t0 = time()
        left, right = 0, len(segs)-1
        mid = (left + right)//2
        if self._check_intersect(l, segs[left]) != 'low':
            print( f'no intersection with upper part for {self._rep_line(l)}. Line is above.')
            return
        if self._check_intersect(l, segs[right]) != 'high':
            print(f'no intersection with upper part for {self._rep_line(l)}. Line is below.')
            return

        while self._check_intersect(l, segs[mid]) != True:
            if verbose: print(f'right now the line intersects with seg {mid} is {self._check_intersect(l, segs[mid])}')
            if self._check_intersect(l, segs[mid]) == 'low':
                left = mid
                mid = (left + right)//2
            else:
                right = mid
                mid = (left + right) // 2
            if verbose: print(f'left is {left} mid is {mid} right is {right}')

        # print(f'Intersect at seg index {mid} {self._rep_seg(segs[mid])}')
        # print('binary search takes {:.3f}.'.format(time() - t0))

    def plot_segs(self, static = False):
        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        new_segs = []
        for seg in self.segs + self.default_segs:
            new_seg = [seg[0], seg[1]]
            new_segs.append(new_seg)
        if static: return new_segs, self.line
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
    """ generate junction(type-1) points
        return a list of 2-tuples
    """
    np.random.seed(seed)
    p = (0, 20)
    juncs = [p]
    for i in range(n-1):
        incx, decy = np.random.random(), np.random.random()
        p = (p[0]+incx, p[1]-decy)
        juncs.append(p)
    return juncs


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, type=int, help='num of points')
parser.add_argument("--line_a", default=1.01, type=float, help='y=ax+b')
parser.add_argument("--line_b", default=1, type=float, help='y=ax+b')

if __name__ == '__main__':
    args = parser.parse_args()
    n = args.n

    for i in range(1):
        print(i)
        s = staircase()
        juncs = gen_juncs(100000, seed=i)
        # juncs = [(0, 10),(1,9),(3,5), (2,8)]
        s.build_segs_from_juncs(juncs)
        # s.sort_segs()

        for k in range(3):
            np.random.seed(k)
            a = 1.01
            b = np.random.random() * (-1e5)
            print(f'{k}: line y = {a} * x + {b}')
            s.find_intersect((a, b), aug=False)
            s.find_intersect_binary((a, b), aug=False)
            print()

        s.plot_segs()

