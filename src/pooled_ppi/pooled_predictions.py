
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

'''
def read_filldata_summary_confidences(zip_json):
    cols = ['fraction_disordered', 'has_clash', 'iptm', 'ptm', 'ranking_score', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm']
    zip_path, json_path = zip_json
    with zipfile.ZipFile(zip_path, 'r') as zip_handle:
        js = json.load(zip_handle.open(json_path))
        s_ = pd.Series(list(js[col] for col in cols), index=cols)
    return s_

def read_filldata(path='alphafold3_predictions', ids=None, include_all_models=False, seed=4):
    if ids is None:
        ids, = snakemake.io.glob_wildcards(os.path.join(path, '{id}.zip'))

    df_ = pd.DataFrame({'id': ids})
    df_['zip'] = df_['id'].map(lambda id: os.path.join(path, f'{id}.zip'))
    df_['zip_size'] = df_['zip'].map(os.path.getsize)
    print(len(df_), 'before filter')
    df_ = df_.query('zip_size > 232').reset_index(drop=True)
    print(len(df_), 'after size filter')

    df_['summary_confidences'] = df_['id'].map(lambda id: f'{id}/{id}_summary_confidences.json')
    df_summary_confidences = pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['summary_confidences'])))
    df_ = pd.concat([df_, df_summary_confidences], axis=1)

    if include_all_models:
        #df_['model0_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-0/summary_confidences.json')
        #df_['model1_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-1/summary_confidences.json')
        #df_['model2_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-2/summary_confidences.json')
        #df_['model3_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-3/summary_confidences.json')
        #df_['model4_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-4/summary_confidences.json')
        # Paths changed in b78e215?
        df_['model0_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-0/{id}_seed-{seed}_sample-0_summary_confidences.json')
        df_['model1_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-1/{id}_seed-{seed}_sample-1_summary_confidences.json')
        df_['model2_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-2/{id}_seed-{seed}_sample-2_summary_confidences.json')
        df_['model3_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-3/{id}_seed-{seed}_sample-3_summary_confidences.json')
        df_['model4_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-4/{id}_seed-{seed}_sample-4_summary_confidences.json')
        df_model_summary_confidences = pd.concat([
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model0_summary_confidences']))).add_prefix('model0_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model1_summary_confidences']))).add_prefix('model1_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model2_summary_confidences']))).add_prefix('model2_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model3_summary_confidences']))).add_prefix('model3_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model4_summary_confidences']))).add_prefix('model4_'),
        ], axis=1)
        df_ = pd.concat([df_, df_model_summary_confidences], axis=1)

    return df_

def chain_pair_iptm_01(s):
    # df_summary_['chain_pair_iptm'].map(af2genomics.alphafold3.chain_pair_iptm_01)
    return json.loads(s)[0][1]

def chain_pair_iptm_triu(s):
    if type(s) is str:
        arr = np.array(json.loads(s))
    elif type(s) is list:
        arr = np.array(s)
    else:
        arr = s
    tri = np.triu_indices_from(arr, k=1)
    return arr[tri]

def parse_pair_iptms(df_summary):
    def interactions_(s):
        l_ = list(itertools.combinations(s.split('_'), 2))
        #random.shuffle(l_)
        return l_

    df_summary['ids'] = df_summary['pool_id'].map(interactions_)
    for model in range(5):
        df_summary[f'model{model}_chain_pair_iptm_triu'] = df_summary[f'model{model}_chain_pair_iptm'].map(chain_pair_iptm_triu)

    iptm_cols_ = [
        'model0_chain_pair_iptm_triu',
        'model1_chain_pair_iptm_triu',
        'model2_chain_pair_iptm_triu',
        'model3_chain_pair_iptm_triu',
        'model4_chain_pair_iptm_triu',
    ]
    df_pairs = df_summary.explode(['ids',] + iptm_cols_).reset_index(drop=True)
    df_pairs['pair_iptm_mean'] = df_pairs[iptm_cols_].mean(axis=1)
    df_pairs['pair_iptm_max'] = df_pairs[iptm_cols_].max(axis=1)
    return df_pairs

def read_summary_confidences(path, cols = ['fraction_disordered', 'has_clash', 'iptm', 'ptm', 'ranking_score', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm']):
    with gzip.open(path, 'r') as handle:
        js = json.load(handle)
        s_ = pd.Series(list(js[col] for col in cols), index=cols)
        return s_

def read_predictions(path='alphafold3_predictions', ids=None, include_all_models=False, seed=4):
    if ids is None:
        ids, = snakemake.io.glob_wildcards(os.path.join(path, '{id}/{id}_model.cif.gz'))

    df_ = pd.DataFrame({'id': ids})
    df_['model'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/{id}_model.cif.gz'))
    df_['summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/{id}_summary_confidences.json.gz'))
    df_summary_confidences = pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['summary_confidences']))
    df_ = pd.concat([df_, df_summary_confidences], axis=1)

    if include_all_models:
        df_['model0_summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/seed-{seed}_sample-0/summary_confidences.json.gz'))
        df_['model1_summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/seed-{seed}_sample-1/summary_confidences.json.gz'))
        df_['model2_summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/seed-{seed}_sample-2/summary_confidences.json.gz'))
        df_['model3_summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/seed-{seed}_sample-3/summary_confidences.json.gz'))
        df_['model4_summary_confidences'] = df_['id'].map(lambda id: os.path.join(path, f'{id}/seed-{seed}_sample-4/summary_confidences.json.gz'))
        df_model_summary_confidences = pd.concat([
            pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['model0_summary_confidences'])).add_prefix('model0_'),
            pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['model1_summary_confidences'])).add_prefix('model1_'),
            pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['model2_summary_confidences'])).add_prefix('model2_'),
            pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['model3_summary_confidences'])).add_prefix('model3_'),
            pd.DataFrame.from_records(tqdm_map(read_summary_confidences, df_['model4_summary_confidences'])).add_prefix('model4_'),
        ], axis=1)
        df_ = pd.concat([df_, df_model_summary_confidences], axis=1)

    return df_

def read_filldata_summary_confidences(zip_json):
    cols = ['fraction_disordered', 'has_clash', 'iptm', 'ptm', 'ranking_score', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm']
    zip_path, json_path = zip_json
    with zipfile.ZipFile(zip_path, 'r') as zip_handle:
        js = json.load(zip_handle.open(json_path))
        s_ = pd.Series(list(js[col] for col in cols), index=cols)
    return s_

def read_filldata_predictions(path='alphafold3_predictions', ids=None, include_all_models=False, seed=4):
    if ids is None:
        ids, = snakemake.io.glob_wildcards(os.path.join(path, '{id}.zip'))

    df_ = pd.DataFrame({'id': ids})
    df_['zip'] = df_['id'].map(lambda id: os.path.join(path, f'{id}.zip'))
    df_['summary_confidences'] = df_['id'].map(lambda id: f'{id}/{id}_summary_confidences.json')
    df_summary_confidences = pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['summary_confidences'])))
    df_ = pd.concat([df_, df_summary_confidences], axis=1)

    if include_all_models:
        df_['model0_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-0/summary_confidences.json')
        df_['model1_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-1/summary_confidences.json')
        df_['model2_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-2/summary_confidences.json')
        df_['model3_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-3/summary_confidences.json')
        df_['model4_summary_confidences'] = df_['id'].map(lambda id: f'{id}/seed-{seed}_sample-4/summary_confidences.json')
        df_model_summary_confidences = pd.concat([
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model0_summary_confidences']))).add_prefix('model0_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model1_summary_confidences']))).add_prefix('model1_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model2_summary_confidences']))).add_prefix('model2_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model3_summary_confidences']))).add_prefix('model3_'),
            pd.DataFrame.from_records(tqdm_map(read_filldata_summary_confidences, zip(df_['zip'], df_['model4_summary_confidences']))).add_prefix('model4_'),
        ], axis=1)
        df_ = pd.concat([df_, df_model_summary_confidences], axis=1)

    return df_

def chain_pair_iptm_01(s):
    # df_summary_['chain_pair_iptm'].map(af2genomics.alphafold3.chain_pair_iptm_01)
    return json.loads(s)[0][1]

def chain_pair_iptm_triu(s):
    if type(s) is str:
        arr = np.array(json.loads(s))
    elif type(s) is list:
        arr = np.array(s)
    else:
        arr = s
    tri = np.triu_indices_from(arr, k=1)
    return arr[tri]

def parse_pair_iptms(df_summary):
    def interactions_(s):
        l_ = list(itertools.combinations(s.split('_'), 2))
        #random.shuffle(l_)
        return l_

    df_summary['ids'] = df_summary['pool_id'].map(interactions_)
    for model in range(5):
        df_summary[f'model{model}_chain_pair_iptm_triu'] = df_summary[f'model{model}_chain_pair_iptm'].map(chain_pair_iptm_triu)

    iptm_cols_ = [
        'model0_chain_pair_iptm_triu',
        'model1_chain_pair_iptm_triu',
        'model2_chain_pair_iptm_triu',
        'model3_chain_pair_iptm_triu',
        'model4_chain_pair_iptm_triu',
    ]
    df_pairs = df_summary.explode(['ids',] + iptm_cols_).reset_index(drop=True)
    df_pairs['pair_iptm_mean'] = df_pairs[iptm_cols_].mean(axis=1)
    df_pairs['pair_iptm_max'] = df_pairs[iptm_cols_].max(axis=1)
    return df_pairs

def read_summary_confidences(path, name):
    js = read_input_json(os.path.join(path, name, f'{name}_summary_confidences.json'))
    return js
'''
