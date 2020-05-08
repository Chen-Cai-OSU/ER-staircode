import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc
from matplotlib.widgets import Slider  # import the Slider widget

try:
    from ER.intersection import EXT, staircase
except ModuleNotFoundError:
    from intersection import EXT, staircase

slider_margin = .1  # multiplication
margin = .1
xmin = 100
xmax = -100
ymin = 100
ymax = -100

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--a", default=1.01, type=float, help='y=ax+b')
parser.add_argument("--b", default=1, type=float, help='y=ax+b')

parser.add_argument("--log", action='store_true')

parser.add_argument("--intersect", default=1, type=float, help='y=ax+b')


def ext(x, margin):
    if x > 0:
        return x * (1 + margin)
    else:
        return x * (1 - margin)


def shrink(x, margin):
    if x > 0:
        return x * (1 - margin)
    else:
        return x * (1 + margin)


if __name__ == '__main__':
    args = parser.parse_args()
    stairs = []
    with open("./I_x.pkl", "rb") as f:
        I_x = pickle.load(f)
    n_stair = len(I_x)

    all_segs = []
    for k in range(10):
        s = staircase()
        key = list(I_x.keys())[k]
        juncs = list(I_x[key].items())  # gen_juncs(10, seed=0) # list(I_x) gen_juncs(100000, seed=0)
        juncs = sorted(juncs, key=lambda x: x[0])
        pprint(juncs)
        s.build_segs_from_juncs(juncs)  # juncs = [(0, 10),(1,9),(3,5), (2,8)]

        np.random.seed(k)
        a = 1.01
        b = args.b  # np.random.random() * (-1e5)
        print(f'{k}: line y = {a} * x + {b}')
        # s.find_intersect((a, b), aug=False)
        # s.find_intersect_binary((a, b), aug=False)
        print()
        new_segs, line = s.plot_segs(static=True)

        xmin = juncs[0][0] if juncs[0][0] < xmin else xmin
        xmax = juncs[-1][0] if juncs[-1][0] > xmax else xmax

        ymin = juncs[-1][1] if juncs[-1][1] < ymin else ymin
        ymax = juncs[0][1] if juncs[0][1] > ymax else ymax

        all_segs.append(new_segs)
        stairs.append(s)

    print(xmin, xmax)
    print(ymin, ymax)

    fig, ax = plt.subplots()

    a_init = np.float(ymax - ymin) / np.float(xmax - xmin)
    b_init = (xmax * ymin - ymax * xmin) / (xmax - xmin)

    a_min, a_max = shrink(a_init, slider_margin), ext(a_init, slider_margin)
    b_min, b_max = shrink(b_init, slider_margin), ext(b_init, slider_margin)

    slider_a_ax = plt.axes([0.1, 0.03, 0.8, 0.03])
    slider_b_ax = plt.axes([0.1, 0.00, 0.8, 0.03])

    ax.set_title('y = a * x + b')

    x = np.linspace(min(0, shrink(xmin, margin)), ext(xmax, margin), 500)
    ax.set_xlim(min(0, shrink(xmin, margin)), ext(xmax, margin) + EXT)
    ax.set_ylim(min(b_init, shrink(ymin, margin)), ext(ymax, margin))

    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    line_plot, = ax.plot(x, a_init * x + b_init, 'r')
    a_slider = Slider(slider_a_ax, 'a', a_min, a_max, valinit=a_init)
    b_slider = Slider(slider_b_ax, 'b', b_min, b_max, valinit=b_init)


    # Next we define a function that will be executed each time the value indicated by the slider changes. The variable of this function will be assigned the value of the slider.
    def update_a(a):
        line_plot.set_ydata(a * x + b_slider.val)
        for s in stairs:
            s.find_intersect_binary((a, b_slider.val), aug=False, check=False, verbose=0)
            print()
        print('-' * 100)
        fig.canvas.draw_idle()  # redraw the plot


    def update_b(b):
        line_plot.set_ydata(a_slider.val * x + b)
        # s.find_intersect_binary((a_slider.val, b), aug=False, check=False)
        for s in stairs:
            s.find_intersect_binary((a_slider.val, b), aug=False, check=False, verbose=0)
            print()
        print('-' * 100)
        fig.canvas.draw_idle()


    # viz one staircase
    for new_segs in all_segs:
        lc = mc.LineCollection(new_segs, linewidths=1)
        ax.add_collection(lc)
    # ax.autoscale()

    # print(s.find_intersect_binary((a_slider.val, b_slider.val), aug=False))
    a_slider.on_changed(update_a)
    b_slider.on_changed(update_b)

    plt.show()
