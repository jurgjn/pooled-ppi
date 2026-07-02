
import itertools, math
import numpy as np, scipy as sp, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, sklearn.metrics
import pooled_ppi, pooled_ppi_yeast as yp
import sklearn, sklearn.metrics

from .core import *

def bin_labels(bins):
    for bin_i, bin_j in itertools.pairwise(bins):
        if bin_i == float('-inf'):
            yield f'<{bin_j}'
        elif bin_j == float('inf'):
            yield f'>{bin_i}'
        else:
            yield f'{bin_i} to {bin_j}'

def groupby_frac(frame, group_name, variable_name):
    return frame.groupby(group_name)[variable_name].mean()

def groupby_fisher(frame, group_name, variable_name, alternative=None):
    n_frame_all = len(frame)
    n_frame_val = len(frame.query(f'`{variable_name}`'))
    print(uf(n_frame_all), 'records')
    print(uf(n_frame_val), 'positives')

    def fisher_exact_(x):
        k = sum(x)
        l = n_frame_val - k
        m = len(x) - k
        n = n_frame_all - (m + k + l)

        (statistic, pvalue) = sp.stats.fisher_exact([[k, l], [m, n]], alternative=alternative)
        odds_ratio = sp.stats.contingency.odds_ratio([[k, l], [m, n]], kind='conditional')
        ci = odds_ratio.confidence_interval(confidence_level=0.95)
        return pd.Series([k, l, m, n, statistic, pvalue, odds_ratio.statistic, ci.low, ci.high, math.log2(odds_ratio.statistic)], index=['k', 'l', 'm', 'n', 'statistic', 'pvalue', 'statistic.odds_ratio', 'ci_low', 'ci_high', 'log2_odds_ratio'])

    return frame.groupby(group_name)[variable_name].apply(fisher_exact_).unstack().astype({'k': int, 'l': int, 'm': int, 'n': int})
