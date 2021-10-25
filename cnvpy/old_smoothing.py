import numpy as np
import pandas as pd

def get_pyramid_weighting(gene_window):
    print('warning: this is duplicated code just to avoid cyclic imports')
    ramp = np.linspace(0, 1, gene_window//2).tolist()
    # actually, should be this, the above has a flat part in the center
    # ramp = np.linspace(0, 1, 1+gene_window//2).tolist()[:-1]

    pyramid_weighting = ramp + [1] + ramp[::-1]
    pyramid_weighting = np.array(pyramid_weighting)
    pyramid_weighting = pyramid_weighting / pyramid_weighting.sum()
    return pyramid_weighting


def chrom_window_generator(gene_var, chromosome, window_size=101, offset=2):
    assert window_size % 2 == 1, 'gene window must be odd number'
    q = gene_var.query('chromosome_name==@chromosome').sort_values('start_position')
    for i in range(0, len(q)-window_size, offset):
        genes = q.iloc[i:(i+window_size)].index
        genomic_start = q.loc[genes[0]].start_position
        genomic_end = q.loc[genes[-1]].end_position
        yield i, genes, genomic_start, genomic_end

def smoothed_expression(adata, chromosome, gene_window=101, offset=2):
    """
    smoothing gene expression on a chromosome via a sliding window
    """
    assert gene_window % 2 == 1, 'gene window must be odd number'
    assert adata.var.query('chromosome_name==@chromosome').shape[0] >= gene_window, "gene window is bigger than the chromosome!"

    # constant_weighting = np.ones(gene_window)
    # constant_weighting = constant_weighting / constant_weighting.sum()

    weight_scheme = get_pyramid_weighting(gene_window).reshape(1, -1)

    smoothed = []
    pos = []
    center_gene = []
    genomic_start = []
    genomic_end = []
    # for fast lookup, generate a table of genename -> index(int) in adata.X[,]
    # such that adata[, gene].X == adata.X[:, gene_index[gene]]
    gene_index = {gene: ix for ix, gene in enumerate(adata.var.index)}
    X = adata.X  # this causes alot of overhead (its not a simple attribute lokup when adata is a VIEW!!). Hence do it outside the loop

    # for i in range(0, len(q)-gene_window, offset):
    for i, genes, window_genomic_start, window_genomic_end in chrom_window_generator(adata.var, chromosome, gene_window, offset):
        # genes = q.iloc[i:(i+gene_window)].index

        if False:
            # somehow this indexing on Adata is very slow, lots of overhead with checking cat-variables etc..
            sum_inwindow = np.sum(adata[:, genes].X * weight_scheme, axis=1)
        else:
            # faster, but uglier
            ix = [gene_index[g] for g in genes]
            sum_inwindow = np.sum(X[:, ix] * weight_scheme, axis=1)

        smoothed.append(sum_inwindow)
        pos.append(i)
        center_gene.append(genes[(gene_window-1)//2])

        genomic_start.append(window_genomic_start)
        genomic_end.append(window_genomic_end)

    if len(smoothed) > 1:
        smoothed = np.stack(smoothed, axis=1)
    else:
        smoothed = None

    pos = pd.DataFrame(pos, columns=['start'])
    pos['chromosome_name'] = chromosome
    pos['end'] = pos['start'] + gene_window
    pos['middle'] = pos['start'] + (gene_window - 1) // 2
    pos['middle_gene'] = center_gene
    pos['genomic_start'] = genomic_start
    pos['genomic_end'] = genomic_end

    return pos, smoothed
