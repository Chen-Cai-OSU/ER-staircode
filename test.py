import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors.kde import KernelDensity
import sys
import networkx as nx
import scipy
from Esme.helper.time import timefunction
import seaborn as sns

def ex():
    # https://stackoverflow.com/questions/39735147/how-to-color-matplotlib-scatterplot-using-a-continuous-value-seaborn-color
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    x, y, z = np.random.rand(3, 100)
    cmap = sns.cubehelix_palette(as_cmap=True)

    f, ax = plt.subplots()
    points = ax.scatter(x, y, c=z, s=50)
    f.colorbar(points)
    plt.show()

def viz_matrix(m, title=''):
    # https://stackoverflow.com/questions/42116671/how-to-plot-a-2d-matrix-in-python-with-colorbar-like-imagesc-in-matlab/42116772
    # m = 100 * np.random.random((50,50))
    print('viz matrix of size (%s %s)'%np.shape(m))
    plt.imshow(m)
    plt.colorbar()
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    viz_matrix(None)