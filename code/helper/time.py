import time
import sys
from helper.format import precision_format
import networkx as nx


def time_node_fil(method, threshold=1):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        assert 'g' in kw.keys()
        g = kw['g']
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>threshold:
                print(f'{method.__name__}takes {precision_format(te-ts, 2)}s. graph is {len(g)}/{len(g.edges())}')
        return result
    return timed

def timefunction(method, threshold=.01):
    # https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>threshold:
                print('%s takes %2.2f s' % (method.__name__, (te - ts) ))
        return result
    return timed

def measurer(method, threshold=1):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>threshold:
                print('%s takes %2.2f s' % (method.__name__, (te - ts) ))

        return result, precision_format(te-ts, 2)
    return timed


@timefunction
def h(g = nx.random_geometric_graph(100,0.1)):
    time.sleep(1.1)
    return 'h'

import signal

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def long_function_call(t = 1.5):
    time.sleep(t)

if __name__ == '__main__':
    h(g = nx.random_geometric_graph(200, 0.1))
    # res, t = h()
    # print(res, t)
    sys.exit()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)   # Ten seconds
    long_function_call(t=3)
    try:
        long_function_call(t=3.5)
    except:
        print ("Timed out!")
