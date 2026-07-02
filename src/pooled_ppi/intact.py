
import functools
import pandas as pd
from cached_path import cached_path
from pooled_ppi.core import *

def interaction_id(uniprot_id_1, uniprot_id_2):
    # sorted pair of uniprot_id-s to compare interacting pairs from different sources
    return '_'.join(sorted([uniprot_id_1, uniprot_id_2]))

def read_mutations():
    # https://www.ebi.ac.uk/intact/download/datasets#mutations
    path_ = cached_path('https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv')
    df_ = pd.read_csv(path_, sep='\t', engine='python').rename({'#Feature AC': 'Feature AC'}, axis=1)
    printlen(df_, 'IntAct mutations')
    df_ = df_.query('`Interaction participants` == `Interaction participants`').copy()
    printlen(df_, 'with defined `Interaction participants`')
    df_['Interaction participants'] = df_['Interaction participants'].str.split('|')
    df_['n_interaction_participants'] = [* map(len, df_['Interaction participants']) ]
    df_ = df_.query('`n_interaction_participants` == 2').copy().reset_index(drop=True)
    printlen(df_, 'with exactly two participants')

    df_participants_ = pd.DataFrame(df_['Interaction participants'].to_list(), columns = ['Participant A', 'Participant B'])
    df_ = pd.concat([df_, df_participants_], axis=1)
    printlen(df_, 'after parsing participants')

    organism_ = '559292 - Saccharomyces cerevisiae'
    df_ = df_.query("`Affected protein organism` == @organism_")
    printlen(df_, 'after querying for', 'organism')

    prefix_ = 'uniprotkb:'
    suffix_ = '(protein(MI:0326), 559292 - Saccharomyces cerevisiae)'
    df_ = df_.query('`Participant A`.str.startswith(@prefix_) & `Participant A`.str.endswith(@suffix_)').copy().reset_index(drop=True)
    printlen(df_, 'after filtering participant A for uniprotkb/protein')
    df_ = df_.query('`Participant B`.str.startswith(@prefix_) & `Participant B`.str.endswith(@suffix_)').copy().reset_index(drop=True)
    printlen(df_, 'after filtering participant B for uniprotkb/protein')

    df_['Participant A'] = df_['Participant A'].str.removeprefix(prefix_).str.removesuffix(suffix_)
    df_['Participant B'] = df_['Participant B'].str.removeprefix(prefix_).str.removesuffix(suffix_)

    df_[['Feature range(s) A', 'Feature range(s) B']] = df_['Feature range(s)'].str.split('-', expand=True)
    df_ = df_.query('(`Feature range(s) A` == `Feature range(s) B`) & (`Resulting sequence`.str.len() == 1)').copy().reset_index(drop=True)
    printlen(df_, 'single-residue substititions')
    df_['Feature range(s) A'] = pd.to_numeric(df_['Feature range(s) A'], errors='coerce')

    labels_ = { #https://www.ebi.ac.uk/intact/download/datasets#mutations
        'mutation with no effect(MI:2226)': 'neutral',
        'mutation disrupting(MI:0573)': 'disrupt/decrease',
        'mutation disrupting strength(MI:1128)': 'disrupt/decrease',
        'mutation disrupting rate(MI:1129)': 'disrupt/decrease',
        'mutation decreasing(MI:0119)': 'disrupt/decrease',
        'mutation decreasing strength(MI:1133)': 'disrupt/decrease',
        'mutation decreasing rate(MI:1130)': 'disrupt/decrease',
        'mutation causing(MI:2227)': 'cause/increase',
        'mutation increasing(MI:0382)': 'cause/increase',
        'mutation increasing strength(MI:1132)': 'cause/increase',
        'mutation increasing rate(MI:1131)': 'cause/increase',
        'mutation(MI:0118)': 'no data on effect',
    }
    df_['effect'] = df_['Feature type'].map(labels_)
    printlen(df_, 'effects -', ' '.join(f'{k}: {v}' for k, v in df_['effect'].value_counts().items()))

    df_['interaction_id'] = [ interaction_id(r['Participant A'], r['Participant B']) for i, r in df_.iterrows() ]
    df_['resid_id'] = [ f'{r["Original sequence"]}{r["Feature range(s) A"]}' for i, r in df_.iterrows() ]

    def apply_(r):
        return r['Participant A'] + '/' + r['Original sequence'] + str(r['Feature range(s) A']) + r['Resulting sequence']
    df_['variant_id'] = df_.apply(apply_, axis=1)

    cols_ = collections.OrderedDict([
        ('variant_id', 'variant_id'),
        ('Participant A', 'uniprot_id1'),
        ('Feature range(s) A', 'pos'),
        ('Original sequence', 'resid'),
        ('resid_id', 'resid_id'),
        ('Participant B', 'uniprot_id2'),
        ('interaction_id', 'interaction_id'),
        ('effect', 'effect'),
    ])
    df_ = df_[cols_.keys()].rename(cols_, axis=1).reset_index(drop=True)
    df_ = df_.drop_duplicates(keep='first').reset_index(drop=True)
    printlen(df_, 'after de-duplication')
    return df_