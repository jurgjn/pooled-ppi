#!/usr/bin/env python
"""
Generate pools as in https://doi.org/10.1101/2025.07.01.662654 except that interactions are weighted by the product of the protein sizes

For Mgen, this leads to 1,791 pools generated in 29 seconds (vs 2,027 pools)

"""

import argparse, itertools, sys, numpy as np, pandas as pd, tqdm, numba
from pprint import pprint

def eprint(*args, **kwargs): # https://stackoverflow.com/questions/5574702/how-do-i-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)

@numba.jit(nopython=True, nogil=True)
def update_coverage_numba(i, j, sizes, cov, cov_row_sums):
    if cov[i, j] == 0:
        val = sizes[i] * sizes[j]
        cov[i, j] = val
        cov[j, i] = val
        cov_row_sums[i] += val
        cov_row_sums[j] += val
        return val, 1
    return 0.0, 0

@numba.jit(nopython=True, nogil=True)
def update_pool_coverage_numba(pool_ix, sizes, cov, cov_row_sums):
    added_sum = 0.0
    added_count = 0
    for idx_r in range(len(pool_ix)):
        r = pool_ix[idx_r]
        for idx_c in range(idx_r + 1, len(pool_ix)):
            c = pool_ix[idx_c]
            s, c_inc = update_coverage_numba(r, c, sizes, cov, cov_row_sums)
            added_sum += 2 * s
            added_count += c_inc
    return added_sum, added_count

@numba.jit(nopython=True, parallel=True, nogil=True)
def pool_expand(pool, pool_ix, sizes, cov, all_row_sums, cov_row_sums, max_size, current_pool_cov_sum, current_pool_all_sum, pool_size_sum):
    """
    Find protein to expand pool while optimising for replication factor
    Returns -1 if nothing can be added
    """
    n = sizes.shape[0]
    repl = np.full(n, 2.0)
    for i in numba.prange(n):
        if not pool[i]:
            if cov_row_sums[i] < all_row_sums[i]:
                new_pool_size = pool_size_sum + sizes[i]
                if new_pool_size <= max_size:
                    added_cov = 0.0
                    for j_ix in pool_ix:
                        added_cov += cov[i, j_ix]

                    added_all = sizes[i] * pool_size_sum
                    repl[i] = (current_pool_cov_sum + 2 * added_cov) / (current_pool_all_sum + 2 * added_all)

    best_i = np.argmin(repl)
    if repl[best_i] < 2:
        return best_i
    else:
        return -1

def generate_pools(sizes, max_size=5120, skip_pairs=[], rng = np.random.default_rng(seed=4)):
    n = sizes.shape[0]
    total_size_sum = sizes.sum()
    all_row_sums = sizes * (total_size_sum - sizes)

    cov = np.zeros((n, n))
    cov_row_sums = np.zeros(n)
    total_cov_sum = 0.0
    total_all_sum = all_row_sums.sum()

    total_cov_count = 0
    total_all_count = n * (n - 1) // 2

    # Return all interactions above max_size as individual two-protein pools
    if n < 10000:
        ii, jj = np.where(np.triu(np.add.outer(sizes, sizes) > max_size, 1))
        for k in range(len(ii)):
            i, j = ii[k], jj[k]
            yield({i, j}, sizes[i] + sizes[j])
            s, c = update_coverage_numba(i, j, sizes, cov, cov_row_sums)
            total_cov_sum += 2 * s
            total_cov_count += c
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if sizes[i] + sizes[j] > max_size:
                    yield({i, j}, sizes[i] + sizes[j])
                    s, c = update_coverage_numba(i, j, sizes, cov, cov_row_sums)
                    total_cov_sum += 2 * s
                    total_cov_count += c

    for i, j in skip_pairs:
        s, c = update_coverage_numba(i, j, sizes, cov, cov_row_sums)
        total_cov_sum += 2 * s
        total_cov_count += c

    pool = np.zeros(n, dtype=np.bool_) # current pool
    pbar = tqdm.tqdm(total=total_all_count)
    pbar.update(total_cov_count)

    while total_cov_sum < total_all_sum:
        if pool.sum() == 0: # current pool is empty
            # randomly selected protein with incomplete coverage
            avail = np.where(cov_row_sums < all_row_sums)[0]
            if len(avail) == 0: break
            avail_choice = rng.choice(avail, 1).squeeze()
            # add to current pool
            pool[avail_choice] = True
            pool_size_sum = sizes[avail_choice]
            current_pool_cov_sum = 0.0
            current_pool_all_sum = 0.0

        while True:
            pool_ix = np.where(pool)[0]
            best_i = pool_expand(pool, pool_ix, sizes, cov, all_row_sums, cov_row_sums, max_size, current_pool_cov_sum, current_pool_all_sum, pool_size_sum)
            if (best_i >= 0):
                added_cov = 0.0
                for j_ix in pool_ix:
                    added_cov += cov[best_i, j_ix]

                current_pool_cov_sum += 2 * added_cov
                current_pool_all_sum += 2 * sizes[best_i] * pool_size_sum
                pool_size_sum += sizes[best_i]
                pool[best_i] = True # add optimal protein to the current pool
            else:
                # Cannot increase current pool anymore; yield as-is and reset search with an empty pool
                yield(set(pool_ix), pool_size_sum)

                # Update global coverage map
                before_count = total_cov_count
                added_sum, added_count = update_pool_coverage_numba(pool_ix, sizes, cov, cov_row_sums)
                total_cov_sum += added_sum
                total_cov_count += added_count

                pool = np.zeros(n, dtype=np.bool_) # Reset pool
                # Update progress bar
                pbar.update(total_cov_count - before_count)
                break
    pbar.close()

def main():
    parser = argparse.ArgumentParser(
        description="Sample random pools minimising overlap"
    )
    parser.add_argument(
        "--init_pools", 
        "-p", 
        help="Pools to skip"
    )
    parser.add_argument(
        "--max_pool_size", 
        "-s", 
        help="Maximum size for pool",
        default=5120,
        type=int,
    )
    parser.add_argument(
        "--max_pools", 
        "-n", 
        help="Maximum number of pools to sample",
        type=int,
    )
    args = parser.parse_args()
    eprint('--init_pools', args.init_pools)
    eprint('--max_pool_size', args.max_pool_size)
    eprint('--max_pools', args.max_pools)

    proteins = pd.read_csv(sys.stdin, sep=r'\s+', names=['seq_id', 'seq_len'])#.head(1000)
    def get_protein_id(ix):
        proteins_id_col = proteins.columns[0]
        return proteins.loc[ix, proteins_id_col]

    #eprint(proteins)
    
    id_to_ix = proteins.reset_index().set_index(proteins.columns[0])['index'].to_dict()
    #eprint(id_to_ix)

    def to_ix_(s):
        return [* map(lambda id_: id_to_ix[id_], s.split('_')) ]

    skip_pairs = []
    if args.init_pools is not None:
        initial_pools = pd.read_csv(args.init_pools, sep=r'\s+')
        initial_pools['pool_ix'] = [ *map(to_ix_, initial_pools['pool_id']) ]
        initial_pools['pool_ix_pairs'] = [ *map(lambda pool_ix: list(itertools.combinations(pool_ix, 2)), initial_pools['pool_ix'] )]
        #eprint(initial_pools)
        def flatten(xss):
            return [x for xs in xss for x in xs]
        skip_pairs = flatten(initial_pools['pool_ix_pairs'].tolist())
        #pprint(skip_pairs)

    #numba.set_num_threads(64) # as things are, multiple threads slow down the code instead of speeding up..
    eprint(numba.get_num_threads(), 'threads available for numba')
    eprint(len(proteins), 'proteins in input')

    sizes = proteins['seq_len'].values
    pools = pd.DataFrame.from_records(itertools.islice(generate_pools(sizes, max_size=args.max_pool_size, skip_pairs=skip_pairs), args.max_pools), columns=['pool_ixs', 'pool_size'])
    pools['pool_ids'] = pools['pool_ixs'].map(lambda ixs: set(map(get_protein_id, ixs)))
    pools['pool_id'] = pools['pool_ids'].map(lambda ids: '_'.join(sorted(ids)))

    eprint(len(pools), 'pools generated')

    def generate_interactions(ids):
        # Generate all possible interactions between ids
        return set(itertools.combinations(sorted(ids), 2))

    # Sanity check - generate a list of all interactions & count size
    all = set(generate_interactions(range(len(sizes))))
    all_sum = 0
    for i, j in all:
        all_sum += sizes[i] * sizes[j]

    # Sanity check - compare to analytic adhoc formula
    ref_sum = (sum(sizes)**2 - sum(sizes*sizes)) / 2
    assert all_sum == ref_sum

    # Sanity check - generate list of interactions from the pools, compare to reference list
    gen = set()
    gen_sum = 0
    for i, r in pools.iterrows():
        pool_interactions = generate_interactions(r.pool_ixs)
        gen |= pool_interactions
        for i, j in pool_interactions:
            gen_sum += sizes[i] * sizes[j]

    eprint(len(all), 'interactions expected')
    eprint(len(gen), 'interactions across all pools generated')
    eprint(gen == all, 'pools include all possible interactions')
    eprint(gen_sum / all_sum, 'length-weighted redundancy factor across all pools') # Should be proportional to the added runtime from the redundancy in the pools

    pools[['pool_id', 'pool_size']].to_csv(sys.stdout, sep='\t', index=False)

if __name__ == "__main__":
    main()
