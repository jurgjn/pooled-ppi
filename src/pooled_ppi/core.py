
import ast, collections, csv, datetime, functools, glob, gzip, hashlib, inspect, io, itertools, json, math, operator, os, os.path, pickle, random, re, requests, shutil, sqlite3, subprocess, string, sys, warnings, zipfile

import pandas as pd

import tqdm.contrib.concurrent

@functools.cache
def guess_prefix(euler_prefix, local_prefix):
    if os.path.isdir(euler_prefix):
        return euler_prefix
    elif os.path.isdir(local_prefix):
        return local_prefix
    else:
        assert False

def projectpath(path):
    dir_ = guess_prefix('/cluster/project/beltrao/jjaenes', '/Users/jjaenes/euler-home/project')
    return os.path.join(dir_, path)

def workpath(path):
    dir_ = guess_prefix('/cluster/work/beltrao/jjaenes', '/Users/jjaenes/euler-home/work')
    return os.path.join(dir_, path)

def flatten(l):
    return [item for sublist in l for item in sublist]

@functools.cache
def get_max_workers():
    try:
        ntasks = int(os.environ['SLURM_NTASKS'])
        source = 'SLURM_NTASKS'
    except:
        ntasks = int(subprocess.check_output(['nproc', '--all']))
        source = 'nproc --all'
    print(f'Using {ntasks} cores inferred from {source}')
    return ntasks

def parallel_map(fn, *iterables):
    """
    Example that returns multiple columns:
        interface_residues[['ifresid1', 'ifresid2']] = pd.DataFrame(mf.parallel_map(mf.structure.get_ifresid, interface_residues['path']))

    See also:
        https://tqdm.github.io/docs/contrib.concurrent/#process_map
    """
    return tqdm.contrib.concurrent.process_map(fn, *iterables, max_workers=get_max_workers(), chunksize=10)

def parallel_from_records(fn, *iterables, columns):
    parallel_map_ = parallel_map(fn, *iterables)
    # Maybe try & infer columns from iterables (fields argument?)
    return pd.DataFrame.from_records(flatten(parallel_map_), columns=columns)
