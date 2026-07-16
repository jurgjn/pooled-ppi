
import argparse, cached_path, collections, collections.abc, copy, functools, glob, hashlib, itertools, gzip, io, json, os, os.path, re, string, subprocess, sys, zipfile, warnings
from pathlib import Path
from pprint import pprint

import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import Bio, Bio.PDB, Bio.PDB.mmcifio, foldcomp

import af3io, pooled_ppi
from pooled_ppi.core import *

from . import plot
from . import bartolec2019, biogrid, childs2019, intact, lambourne2026, string_db

def path(subpath):
    return Path('/cluster/project/beltrao/jjaenes/25.12_pooled-ppi-yeast') / subpath
 
@functools.cache
def get_data():
    for prefix in map(Path, ['/data', '/workspace/data', '/contents/data', '/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07']):
        if prefix.is_dir():
            printsrc(f'Using data from: {prefix}')
            return prefix

cached_path.set_cache_dir(get_data() / 'resources/.cached_path/')
print(cached_path.get_cache_dir())

def get_path(path):
    return get_data() / path

def get_proteins(cols=None):
    proteins = pd.read_parquet(get_path('proteins.parquet'), columns=cols)
    proteins['protein'] = proteins['uniprot_entry'].str.removesuffix('_YEAST') + '/' + proteins['uniprot_locus'] + '/' + proteins['uniprot_id']
    return proteins

def merge_af3_id(frame, frame_on, proteins_on, suffix='', v=True):
    # proteins_on is either uniprot_id or uniprot_locus
    mapping = get_proteins().set_index(proteins_on)['af3_id'].to_dict()
    af3_id_col = f'af3_id{suffix}'
    frame_ = frame.copy()
    frame_[af3_id_col] = frame_[frame_on].map(mapping)
    frame_ = frame_.query(f'{af3_id_col} == {af3_id_col}').reset_index(drop=True)
    if v: print(f'{ul(frame_)} of {ul(frame)} rows mapped via {frame_on}/{proteins_on}')
    return frame_

def merge_af3_pair(frame, on1, on2, proteins_on, prefix, v=True):
    frame_ = frame.copy()
    frame_ = merge_af3_id(frame_, on1, proteins_on, suffix='1', v=v)
    frame_ = merge_af3_id(frame_, on2, proteins_on, suffix='2', v=v)
    frame_['af3_pair'] = np.where(
        frame_['af3_id1'] < frame_['af3_id2'],
        frame_['af3_id1'] + '_' + frame_['af3_id2'],
        frame_['af3_id2'] + '_' + frame_['af3_id1'],
    )
    return frame_.set_index('af3_pair').add_prefix(f'{prefix}:')

def get_head(path, nrows=1):
    return next(pq.ParquetFile(get_path(path)).iter_batches(batch_size=nrows)).to_pandas().transpose()

def get_pairs(sel=None, cols=None, cols_proteins=['af3_id', 'uniprot_id', 'uniprot_locus', 'uniprot_entry', 'is_expressed', 'seq']):
    pairs = pd.read_parquet(get_path('summary_pairs.parquet'), filters=sel, columns=cols)\
        .merge(get_proteins(cols_proteins).add_suffix('1'), on='af3_id1')\
        .merge(get_proteins(cols_proteins).add_suffix('2'), on='af3_id2')
    printlen(pairs, f"pairs from {get_path('summary_pairs.parquet')}")
    return pairs

@functools.cache
def get_pairs_all_vs_all():
    return get_pairs(sel=[('isin_all_vs_all', '==', True),])

@functools.cache
def get_pairs_std(sel=None, cols=None, cols_proteins=['af3_id', 'uniprot_id', 'uniprot_locus', 'uniprot_entry', 'is_expressed']):
    cols_metrics_ = (
        'chain_pair_iptm_best',
        'chain_pair_iptm_best_diff',
        'chain_pair_iptm_best_ratio',
        'chain_pair_iptm_mean',
        'chain_pair_iptm_mean_diff',
        'chain_pair_iptm_mean_ratio',
        'ipsae:pDockQ',
        'ipsae:pDockQ2',
        'ipsae:LIS',
        'ipsae:ipSAE',
    )
    cols_pairs_ = ('af3_id1', 'af3_id2', 'name', 'physical:combined_score', ) + cols_metrics_
    cols_proteins_ = ('af3_id', 'uniprot_locus', 'uniprot_entry', 'uniprot_id', 'is_expressed',)
    return get_pairs(cols=None, cols_proteins=cols_proteins_).query('is_expressed1 & is_expressed2').reset_index(drop=True)

@functools.cache
def get_pairs_std_matrix(values):
    cols_ = ['af3_id1', 'af3_id2', values]
    return pd.concat([
        get_pairs_std()[cols_],
        get_pairs_std()[cols_].rename({'af3_id1': 'af3_id2', 'af3_id2': 'af3_id1'}, axis=1)
    ], axis=0).pivot(index='af3_id1', columns='af3_id2', values=values)

@functools.cache
def get_pairs_std_matrix_corr(values, *args, **kwargs):
    frame = get_pairs_std_matrix(values=values).corr(*args, **kwargs)
    frame.index.name = 'af3_id1'
    return frame

@functools.cache
def get_pairs_std_matrix_corr_pairs(values, *args, **kwargs):
    frame = get_pairs_std_matrix_corr(values, *args, **kwargs).reset_index()
    return pd.melt(frame, id_vars=['af3_id1'], value_vars=frame.columns.tolist()[1:], value_name=f'{values}_corr').query('af3_id1 < af3_id2')

def _num_rows(path):
    return pq.ParquetFile(get_path(path)).metadata.num_rows

def get_pairs_nunique():
    return _num_rows(get_path('summary_pairs.parquet'))

def get_pools_nunique():
    return pd.read_parquet(get_path('summary_pairs.parquet'), columns=['name'])['name'].nunique()

def get_pools_total():
    return _num_rows(get_path('pools.parquet'))

def open_predictions_db(ids):
    return foldcomp.open(get_path('predictions-db/predictions-db'), ids=ids)

def load_predictions_db(ids):
    parser = Bio.PDB.PDBParser(QUIET=True)
    struct0 = None
    with open_predictions_db(ids) as db:
        for index, ((name, pdb), chain_id) in enumerate(itertools.islice(zip(db, af3io.input.enumerate_chains()), None)):
            struct = parser.get_structure(index, io.StringIO(pdb))
            if index == 0:
                struct0 = struct
            else:
                chain0 = next(struct[0].get_chains())
                chain0.id = chain_id
                struct0[0].add(chain0)
    return struct0

def save_predictions_db(ids, file):
    struct = load_predictions_db(ids)
    pdbio = Bio.PDB.PDBIO()
    pdbio.set_structure(struct)
    pdbio.save(file)

def read_predictions_db(ids):
    # Fetch PDB-formatted string with the interaction chains from foldcomp
    pdbio = io.StringIO()
    save_predictions_db(ids, pdbio)
    pdbio.seek(0)
    return pdbio.read()

def get_models(filters=None):
    pairs = pd.read_parquet(get_path('summary_models.parquet'), filters=filters)
    pairs['db_id1'] = pairs['input_name_max'] + '_' + pairs['sample_max'].astype(str) + '_' + pairs['chain_id1_max']
    pairs['db_id2'] = pairs['input_name_max'] + '_' + pairs['sample_max'].astype(str) + '_' + pairs['chain_id2_max']
    return pairs

def get_pair_pdb(pairs_row):
    key1 = f'{pairs_row.name.squeeze()}_{pairs_row.af3_id1.squeeze()}'
    key2 = f'{pairs_row.name.squeeze()}_{pairs_row.af3_id2.squeeze()}'
    return load_predictions_db([key1, key2])

def sample_controls(pairs, col_pos, col_neg, neg_ratio=1000, exclude_homodimers=True):
    if exclude_homodimers:
        pairs = pairs.query('af3_id1 != af3_id2')
    n_pos = len(pairs.query(col_pos))
    return pd.concat([
        pairs.query(col_pos).assign(is_positive=True),
        pairs.query(col_neg).sample(neg_ratio*n_pos, random_state=pooled_ppi.GUARANTEED_RANDOM).assign(is_positive=False),
    ]).reset_index(drop=True)

'''
class PooledPredictionsDb:
    def __init__(self, path='/data', columns_proteins=['af3_id', 'uniprot_id', 'uniprot_locus', 'uniprot_entry'], columns_pairs=['af3_pair', 'af3_id1', 'af3_id2', 'name', 'physical:combined_score', 'chain_pair_iptm_mean_corrected']):
        self.path = path
        self.proteins = pd.read_parquet(os.path.join(self.path, 'proteins.parquet'), columns=columns_proteins)
        self.pairs = pd.read_parquet(os.path.join(self.path, 'summary_pairs.parquet'), columns=columns_pairs)\
            .merge(self.proteins.add_suffix('1'), on='af3_id1')\
            .merge(self.proteins.add_suffix('2'), on='af3_id2')
        print(f'{ul(self.pairs)} pairs / {uf(self.pairs["name"].nunique())} pools')

    def bait_prey(self):
        pairs_fwd = self.pairs.copy()
        pairs_fwd['bait_id'] = pairs_fwd['uniprot_id1']
        pairs_fwd['prey_id'] = pairs_fwd['uniprot_id2']

        pairs_rev = self.pairs.copy()
        pairs_rev['bait_id'] = pairs_rev['uniprot_id2']
        pairs_rev['prey_id'] = pairs_rev['uniprot_id1']

        bait_prey = pd.concat([pairs_fwd, pairs_rev], axis=0)
        return bait_prey

    def save_ids(self, ids, file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        struct0 = None
        with foldcomp.open(os.path.join(self.path, 'predictions-db/predictions-db'), ids=ids) as db:
            for index, ((name, pdb), chain_id) in enumerate(itertools.islice(zip(db, af3io.input.enumerate_chains()), None)):
                struct = parser.get_structure(index, io.StringIO(pdb))
                if index == 0:
                    struct0 = struct
                else:
                    chain0 = next(struct[0].get_chains())
                    chain0.id = chain_id
                    struct0[0].add(chain0)
                
        pdbio = Bio.PDB.PDBIO()
        pdbio.set_structure(struct0)
        pdbio.save(file)
'''