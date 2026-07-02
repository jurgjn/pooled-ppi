
import functools
import pandas as pd
from cached_path import cached_path
from pooled_ppi.core import *

def read(*args, **kwargs):
    path = cached_path('/cluster/project/beltrao/jjaenes/25.12_pooled-ppi-yeast/data/resources/Bartolec2019/ac9b03975_si_002.xlsx')
    frame = pd.read_excel(path, *args, **kwargs)
    return frame
