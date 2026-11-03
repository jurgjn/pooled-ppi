
import importlib

from .core import *

# Load submodules lazily so lightweight entry points (e.g. pooled-ppi-sample) don't import heavy dependencies or trigger side effects
_submodules = {'fasta', 'idmapping', 'ipsae', 'molstar', 'predictions', 'predictions_db', 'size_correction', 'stats', 'structure'}
_attrs = {'PooledPredictions': 'predictions', 'PooledPredictionsDb': 'predictions_db'}

def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f'.{name}', __name__)
    if name in _attrs:
        return getattr(importlib.import_module(f'.{_attrs[name]}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__():
    return sorted(set(globals()) | _submodules | set(_attrs))
