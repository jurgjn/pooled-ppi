
"""
https://www.nature.com/articles/s41467-026-70942-x
"""

import pandas as pd

from cached_path import cached_path

from pooled_ppi.core import *

def read_MOESM(n):
    path = cached_path(f'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-70942-x/MediaObjects/41467_2026_70942_MOESM{n}_ESM.xlsx')
    frame = pd.read_excel(path)
    printlen(frame, f'in 41467_2026_70942_MOESM{n}_ESM.xlsx')
    return frame

def read_scPRS_v2():
    print('final scPRS-v2 of 108 PPIs')
    path = cached_path('https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-70942-x/MediaObjects/41467_2026_70942_MOESM3_ESM.xlsx')
    frame = pd.read_excel(path)
    return frame

def read_scRRS_v2():
    print('final scRRS-v2 of 198 pairs')
    path = cached_path('https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-70942-x/MediaObjects/41467_2026_70942_MOESM4_ESM.xlsx')
    frame = pd.read_excel(path)
    return frame
