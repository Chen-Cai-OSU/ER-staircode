import numpy as np
import matplotlib.pyplot as plt

def hist_plot(d, show = False):
    """
    :param d:
    :return:
    """
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Density')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    d = np.random.laplace(loc=15, scale=3, size=500).reshape(500, 1)
    print(d.shape)
    hist_plot(d)