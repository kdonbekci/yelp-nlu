import os
import numpy as np
import pandas as pd
import pickle
import itertools
from scipy.spatial.distance import cosine

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
GLOVE_DIR = os.path.join(DATA_DIR, 'glove.6B')
MODELS_DIR = os.path.join(SRC_DIR, '..', 'models')
UNK_KEY = '<UNK>'
NULL_KEY = '<NULL>'
YELP_DIR = os.path.join(DATA_DIR, 'yelp')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
MISC_DIR = os.path.join(DATA_DIR, 'misc')

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def find_subsets_of_n(S,n):
    if n == 0:
        return []
    return list(itertools.combinations(S, n)) #+ find_subsets_up_to_n(S, n-1)


def glove2dict(src_filename):
    """ by Christopher Potts
    GloVe Reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    """
    data = {}
    with open(src_filename) as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def neighbors(w, lookup, distfunc=cosine):
    """Tool for finding the nearest neighbors of `word` in `df` according
    to `distfunc`. The comparisons are between row vectors.

    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.
    df : pd.DataFrame
        The vector-space model.
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If word is not in `df.index`.

    Returns
    -------
    pd.Series
        Ordered by closeness to `word`.
    """
    dists = pd.DataFrame([{word:distfunc(w, v) for word, v in lookup.items()}])
    return dists, np.argsort(dists.to_numpy()[0])

def load_data(lookup):
    data  = {}
    for key in lookup:
        data[key] = {}
        for c in lookup[key]:
            path = os.path.join(DATA_DIR, 'dataset', '{}-{}.npy'.format(key, c))
            if c == 'text':
                x = np.load(path, allow_pickle=True)
            else:
                x = np.load(path, allow_pickle=False)
            data[key][c] = x
    return data

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
def save_pickle(path, data):
    with open(path, 'wb+') as f:
        pickle.dump(data, f)
        
