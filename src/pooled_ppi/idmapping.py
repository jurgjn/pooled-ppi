
import functools, requests
from cached_path import cached_path

from .core import *

def read_sec_ac():
    path = cached_path('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/sec_ac.txt')
    # https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/delac_sp.txt
    # https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/delac_tr.txt
    return pd.read_csv(path, skiprows=31, names=['Secondary AC', 'Primary AC'], sep=r"\s+",)

def read_idmapping():
    idmapping = pd.read_csv(
        cached_path('https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/idmapping/by_organism/YEAST_559292_idmapping.dat.gz'),
        #cached_path('https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/idmapping/by_organism/YEAST_559292_idmapping_selected.tab.gz'),
        compression='gzip',
        sep='\t',
        names=['UniProtKB_AC', 'ID_type', 'ID'],
    )
    return idmapping

@functools.cache
def read_mapping():
    # Add other-primary
    idmapping = read_idmapping()

    # Add secondary-primary
    sec_ac = read_sec_ac().rename({
        'Secondary AC': 'ID',
        'Primary AC': 'UniProtKB_AC'
    }, axis=1)
    sec_ac.insert(loc=1, column='ID_type', value='sec_ac')
    sec_ac = sec_ac[['UniProtKB_AC', 'ID_type', 'ID']]

    # Add primary-primary
    primary_ac = pd.DataFrame({'UniProtKB_AC': sec_ac['UniProtKB_AC'].drop_duplicates(keep='first')})
    primary_ac['ID_type'] = 'primary_ac'
    primary_ac['ID'] = primary_ac['UniProtKB_AC']

    return pd.concat([idmapping, sec_ac, primary_ac,], axis=0)\
        .groupby(['UniProtKB_AC', 'ID'], sort=False)['ID_type'].agg(lambda s: ','.join(sorted(s.unique()))).reset_index()\
        [['ID', 'UniProtKB_AC', 'ID_type']]

'''
def query(query=["CASQ2", "CASQ1", "GSTO1", "DMD", "GSTM2"], target='UNIPROTSWISSPROT_ACC', organism='hsapiens', numeric_namespace='ENTREZGENE_ACC'):
    """
    Query for HGNC gene names using g:convert (https://biit.cs.ut.ee/gprofiler/convert)
    """
    r = requests.post(url='https://biit.cs.ut.ee/gprofiler/api/convert/convert/', json=locals())
    df_ = pd.DataFrame(r.json()['result'])
    return df_

@functools.cache
def query_cached(*args, **kwargs):
    return query(*args, **kwargs)

def proxy_mapping(l, r, organism='hsapiens', target='ENSG', v=False):
    """
    Generate unique mappings over a proxy namespace
    """
    both = set(l) & set(r)
    if v: print(f'{ul(l)}\t{ul(r)} {l.name}/{r.name} ({ul(both)} overlapping)')

    l_dedup = l.drop_duplicates(keep='first')
    r_dedup = r.drop_duplicates(keep='first')
    if v: print(f'{ul(l_dedup)}\t{ul(r_dedup)} after dedup')

    l_query = query_cached(tuple(l_dedup.tolist()), organism=organism, target=target)[['incoming', 'converted']].rename({'incoming': l.name, 'converted': target}, axis=1)
    r_query = query_cached(tuple(r_dedup.tolist()), organism=organism, target=target)[['incoming', 'converted']].rename({'incoming': r.name, 'converted': target}, axis=1)
    if v: print(f'{ul(l_query)}\t{ul(r_query)} from query')

    l_query = l_query.query(f'{target} != "None"')
    r_query = r_query.query(f'{target} != "None"')
    if v: print(f'{ul(l_query)}\t{ul(r_query)} after removing empty mappings')

    merge = l_query.merge(r_query, on=target)
    if v: print(f'{ul(merge)} after merge')

    merge = merge[[l.name, r.name]].drop_duplicates(keep='first')
    if v: print(f'{ul(merge)} after shared dedup')

    merge = merge.drop_duplicates(subset=[l.name], keep=False)
    if v: print(f'{ul(merge)} after left dedup')

    merge = merge.drop_duplicates(subset=[r.name], keep=False)
    if v: print(f'{ul(merge)} after right dedup')

    return merge.reset_index(drop=True)

def mapping(l, r, organism='hsapiens', target='ENSG', v=False):
    both = set(l) & set(r)
    if v: print(f'{ul(both)}\toverlapping')

    l_uniq = pd.Series(list(set(l) - both), name=l.name)
    r_uniq = pd.Series(list(set(r) - both), name=r.name)
    #print(r_uniq)

    mapping_both = pd.DataFrame({
        l.name: sorted(both),
        r.name: sorted(both),
    })
    mapping_uniq = proxy_mapping(l_uniq, r_uniq, organism=organism, target=target, v=v)
    return pd.concat([mapping_both, mapping_uniq], axis=0)

def proxy_merge(left, right, left_on, right_on, right_prefix=None, organism='hsapiens', target='ENSG', v=False):
    mapping = proxy_mapping(left[left_on], right[right_on], organism, target, v)
    merged = left.merge(mapping, on=left_on, how='left').merge(right, on=right_on, how='left').reset_index(drop=True)
    if right_prefix is not None:
        mapper_ = {col: col if col.startswith(right_prefix) else right_prefix + col for col in right.columns}
        merged = merged.rename(mapper_, axis=1)
    return merged
'''