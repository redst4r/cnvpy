import numpy as np
import tqdm
import pandas as pd
from anndata import AnnData
from sctools.scplotting import godsnot_64
import fastcluster
import seaborn as sns
from scipy.sparse import issparse
import matplotlib.patches as mpatches

CHROMOSOMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', 'X']


def filter_genes(A, B, min_cells):
    cellsA = np.sum(A.X > 0, axis=0)  # cells expressing the gene
    cellsB = np.sum(B.X > 0, axis=0)  # cells expressing the gene

    cellsA = np.array(cellsA).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array
    cellsB = np.array(cellsB).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array

    ix = np.logical_and(cellsA > min_cells,
                        cellsB > min_cells)

    return A.var.iloc[ix].index


def get_pyramid_weighting(gene_window):
    pyramid_weighting = np.linspace(0, 1, gene_window//2).tolist() + [1] + np.linspace(0, 1, gene_window//2).tolist()[::-1]
    pyramid_weighting = np.array(pyramid_weighting)
    pyramid_weighting = pyramid_weighting / pyramid_weighting.sum()
    return pyramid_weighting


def smoothed_expression(adata, chromosome, gene_window=101, offset=2):
    """
    smoothing gene expression on a chromosome via a sliding window
    """
    assert gene_window % 2 == 1, 'gene window must be odd number'
    # constant_weighting = np.ones(gene_window)
    # constant_weighting = constant_weighting / constant_weighting.sum()

    weight_scheme = get_pyramid_weighting(gene_window).reshape(1, -1)

    q = adata.var.query('chromosome_name==@chromosome').sort_values('start_position')
    smoothed = []
    pos = []
    for i in range(0, len(q)-gene_window, offset):
        genes = q.iloc[i:(i+gene_window)].index
#         sum_inwindow = adata[:,genes.symbol].X.mean(axis=1).toarray()
        sum_inwindow = np.sum(adata[:, genes].X * weight_scheme, axis=1)
        smoothed.append(sum_inwindow)
        pos.append(i)
    if len(smoothed) > 1:
        smoothed = np.stack(smoothed, axis=1)
    else:
        smoothed = None

    pos = pd.DataFrame(pos, columns=['position'])
    pos['chromosome_name'] = chromosome
    return pos, smoothed


def smoothed_expression_all_chromosomes(adata, gene_window=101):
    """
    smoothing gene expression on a chromosome via a sliding window
    """
    X = []
    pos = []
    for chromosome in tqdm.tqdm(CHROMOSOMES):
        pos_df, s = smoothed_expression(adata, chromosome, gene_window)
        if isinstance(s, np.ndarray):  # not sure what htat is
            X.append(s)
            pos.append(pos_df)
    X = np.concatenate(X, axis=1)
    pos = pd.concat(pos)
    return X, pos  # , chrom_indicator


def _preprocess(adata, ref_field, ref_groups):
    """
    - splitting into reference samples and tumor samples
    - centering on the reference
    - clipping
    """
    Qnormal = adata[adata.obs[ref_field].isin(ref_groups)]
    Qtumor = adata[~adata.obs[ref_field].isin(ref_groups)]

    gene_ix = filter_genes(Qnormal, Qtumor, 10)
    Qnormal = Qnormal[:, gene_ix]
    Qtumor = Qtumor[:, gene_ix]

    # center based on the reference, before smoothing!
    if issparse(adata.X):
        Qnormal.X = Qnormal.X.A
        Qtumor.X = Qtumor.X.A

    ref_mean = Qnormal.X.mean(0, keepdims=True)
    Qnormal.X = Qnormal.X - ref_mean
    Qtumor.X = Qtumor.X - ref_mean

    # clipping
    Qnormal.X = np.clip(Qnormal.X, -3, 3)
    Qtumor.X = np.clip(Qtumor.X, -3, 3)

    return Qnormal, Qtumor


def _postprocess(smoothed_normal, smoothed_tumor):
    """
    after smoothing on the chromosome:
    - center each cell at median
    - center against reference
    - undo the log
    """

    # center each cell at its median expression (assumption: most genes wont be CNV)
    smoothed_tumor = smoothed_tumor - np.median(smoothed_tumor, axis=1, keepdims=True)
    smoothed_normal = smoothed_normal - np.median(smoothed_normal, axis=1, keepdims=True)

    # relative to the normal cells
    relative_tumor = smoothed_tumor - smoothed_normal.mean(0, keepdims=True)
    relative_normal = smoothed_normal - smoothed_normal.mean(0, keepdims=True)

    # ubdo log
    exp_relative_tumor = np.exp(relative_tumor)
    exp_relative_normal = np.exp(relative_normal)

    return exp_relative_normal, exp_relative_tumor


def my_inferCNV(adata, ref_field, ref_groups, verbose=True):

    # Note: the .values is important otherwise the `in` doesnt work!?
    assert all([g in adata.obs[ref_field].values for g in ref_groups]), f"some groups dont exist in {ref_field}"

    # check if is a count matrix: we want lognormlaized!
    d = adata.X-adata.X.astype(int)
    if issparse(adata.X):
        if d.nnz == 0:
            raise ValueError('Data should be normlaized and logtransformed!')
    else:
        if np.all(d == 0):
            raise ValueError('Data should be normlaized and logtransformed!')

    if verbose:
        print('Preprocessing')
    Qnormal, Qtumor = _preprocess(adata, ref_field, ref_groups)
    if verbose:
        print('smoothing Tumor')
    smoothed_tumor, tumor_chr = smoothed_expression_all_chromosomes(Qtumor)
    if verbose:
        print('smoothing normal')
    smoothed_normal, normal_chr = smoothed_expression_all_chromosomes(Qnormal)

    if verbose:
        print('postprocessing')
    exp_relative_normal, exp_relative_tumor = _postprocess(smoothed_normal, smoothed_tumor)

    CNV_TUMOR = AnnData(exp_relative_tumor, obs=Qtumor.obs, var=tumor_chr)
    CNV_NORMAL = AnnData(exp_relative_normal, obs=Qnormal.obs, var=normal_chr)

    return CNV_NORMAL, CNV_TUMOR


def plotting(S: AnnData, row_color_fields):

    if not isinstance(row_color_fields, list):
        row_color_fields = [row_color_fields]

    chrom_colormap = {c: godsnot_64[i] for i, c in enumerate(CHROMOSOMES)}
    chrom_colors = [chrom_colormap[i] for i in S.var.chromosome_name]

    color_df = []
    colormaps = {}
    for f in row_color_fields:
        types = S.obs[f].unique()
        colormap = {ct: godsnot_64[i] for i, ct in enumerate(types)}
        color_vector = S.obs[f].apply(lambda x: colormap[x])
        color_df.append(color_vector)
        colormaps[f]= colormap
        # celltype_colors = [colormap[ct] for ct in S.obs[row_color_field]]
    color_df = pd.DataFrame(color_df).T

    print('Clustering')
    linkage_s = fastcluster.linkage(S.X, method='ward')

    print('Drawing')
    X = pd.DataFrame(S.X, index=S.obs.index)
    g = sns.clustermap(X, col_cluster=False, cmap="bwr",
                       vmin=0.5, vmax=1.5,
                       row_linkage=linkage_s,
                       col_colors=chrom_colors,
                       row_colors=color_df,
                       xticklabels=False,
                       yticklabels=False
                       )

    leg = []
    for f in row_color_fields:
        legend_TN = [mpatches.Patch(color=color, label=ct) for ct, color in colormaps[f].items()]
        leg.extend(legend_TN)

    l2 = g.ax_heatmap.legend(loc='center left', bbox_to_anchor=(1.01, 0.85),
                             handles=leg, frameon=True)
    l2.set_title(title=f, prop={'size': 10})

    return g
