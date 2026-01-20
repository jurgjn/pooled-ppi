
import glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path

import numpy as np, pandas as pd, snakemake.io

import af3io

from .core import *

class PooledPredictions:
    def __init__(self, path):
        # find/assign name (from path)
        self.path = Path(path)
        ids, = snakemake.io.glob_wildcards(os.path.join(path, '{id}.zip'))
        self.predictions = pd.DataFrame({'id': ids})
        self.predictions['zip'] = self.predictions['id'].map(lambda id: os.path.join(path, f'{id}.zip'))
        self.summary_confidences = pd.concat(parallel_map(af3io.predictions.read_summary_confidences, self.predictions['zip']), axis=0).reset_index(drop=True)

def read_summary_confidences(path):
    pp = PooledPredictions(path)
    return pp.summary_confidences

def chain_pair_iptm_triu(s):
    if type(s) is str:
        arr = np.array(json.loads(s))
    elif type(s) is list:
        arr = np.array(s)
    else:
        arr = s
    tri = np.triu_indices_from(arr, k=1)
    return arr[tri]

def explode_iptms(pools, columns_triu=['chain_pair_iptm']):
    def interactions_(s):
        l_ = list(itertools.combinations(s.split('_'), 2))
        #random.shuffle(l_)
        return l_

    pairs = pd.DataFrame({'ids': pools['pool_id'].map(interactions_)})
    for column in columns_triu:
        pairs[column] = pools[columns_triu].map(chain_pair_iptm_triu)

    return pairs.explode(['ids',] + columns_triu).reset_index(drop=True)
