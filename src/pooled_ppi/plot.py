
import numpy as np, scipy as sp, pandas as pd, matplotlib, matplotlib.pyplot as plt, seaborn as sns, sklearn, sklearn.metrics

from brokenaxes import brokenaxes
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles

def summary_pairs_venn2(data_, queries, title=None):
    venn2((
        set(data_.query(queries[0])['af3_pair']),
        set(data_.query(queries[1])['af3_pair']),
        ),
        set_labels = (
            queries[0],
            queries[1],
        ),
    )
    if title is not None:
        plt.gca().set_title(title)

def summary_pairs_venn3(data_, queries, title=None):
    venn3((
        set(data_.query(queries[0])['af3_pair']),
        set(data_.query(queries[1])['af3_pair']),
        set(data_.query(queries[2])['af3_pair']),
        ),
        set_labels = (
            queries[0],
            queries[1],
            queries[2],
        ),
    )
    if title is not None:
        plt.gca().set_title(title)

def ax_format_uf(ax):
    ax.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
