
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

def agg_chain_pair_iptms(frame):
    """Aggregate pairwise iptms (chain_pair_iptm) using different approaches:
        chain_pair_iptm_best = best-ranked sample
        chain_pair_iptm_max = max across samples
        chain_pair_iptm_mean = mean across samples
    """
    # np.array([np.array([[1,2], [3,4]]), np.array([[4,3], [2,1]]), np.array([[1,1], [1,1]])]).max(axis=0)
    chain_pair_iptm_best = as_array(frame.sort_values('ranking_score', ascending=False).head(1)['chain_pair_iptm'].squeeze())

    chain_pair_iptm_list = list(frame['chain_pair_iptm'].map(as_array))
    chain_pair_iptm_max = np.maximum.reduce(chain_pair_iptm_list)

    chain_pair_iptm_array = np.array(chain_pair_iptm_list)
    chain_pair_iptm_mean = chain_pair_iptm_array.mean(axis=0)

    return pd.Series([chain_pair_iptm_best, chain_pair_iptm_max, chain_pair_iptm_mean], index=['chain_pair_iptm_best', 'chain_pair_iptm_max', 'chain_pair_iptm_mean'])

def explode_iptms(pools, columns_triu=['chain_pair_iptm']):
    def interactions_(s):
        l_ = list(itertools.combinations(s.split('_'), 2))
        #random.shuffle(l_)
        return l_

    pairs = pd.DataFrame({'ids': pools['pool_id'].map(interactions_)})
    for column in columns_triu:
        pairs[column] = pools[column].map(chain_pair_iptm_triu)

    return pairs.explode(['ids',] + columns_triu).reset_index(drop=True)
