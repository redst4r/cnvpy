import numpy as np
import tqdm
import pandas as pd
from anndata import AnnData
from sctools.scplotting import godsnot_64
import fastcluster
import seaborn as sns
from scipy.sparse import issparse
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import cut_tree

CHROMOSOMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', 'X']


class inferCNV():

    def __init__(self, verbose=True):
        self.CNV_NORMAL = None  # where we store the smoothed data
        self.CNV_TUMOR = None
        self.linkage_normal = None
        self.linkage_tumor = None
        self.verbose = verbose

    def infer(self, adata, ref_field, ref_groups):
        """
        running the inferCNV method, basically smoothing along the chromosomes
        """
        self.CNV_NORMAL, self.CNV_TUMOR = my_inferCNV(
            adata,
            ref_field,
            ref_groups,
            verbose=self.verbose)

    def cluster(self):
        """
        cluster the CNV profiles of both N/T
        """
        self.linkage_normal = fastcluster.linkage(self.CNV_NORMAL.X, method='ward')
        self.linkage_tumor = fastcluster.linkage(self.CNV_TUMOR.X, method='ward')

    def plotting(self, row_color_fields, which, denoise=True, vmin=0.5, vmax=1.5):
        """
        plot the heatmap of the CNV profiles of either N or Tv
        """
        assert which in ['normal', 'tumor']
        assert self.linkage_normal is not None and self.linkage_tumor is not None, "not clustered yet, run .cluster()"
        S = self.CNV_NORMAL if which == 'normal' else self.CNV_TUMOR
        clustering = self.linkage_normal if which == 'normal' else self.linkage_tumor
        plotting(S, row_color_fields, clustering=clustering, vmin=vmin, vmax=vmax)

    def annotate_clusters(self, n_clusters_normal, n_clusters_tumor):
        """
        truncating the hierachical tree of the CNV clusters.
        Returns dataframe which annotate for each cell the CNV cluster

        """
        N_clusters = cut_tree(self.linkage_normal, n_clusters=n_clusters_normal).flatten().astype('str')
        T_clusters = cut_tree(self.linkage_tumor, n_clusters=n_clusters_tumor).flatten().astype('str')

        df_normal = pd.DataFrame(N_clusters, columns=['CNVcluster'], index=self.CNV_NORMAL.obs.index)
        df_normal['CNVcluster'] = 'N'+df_normal['CNVcluster']
        df_tumor = pd.DataFrame(T_clusters, columns=['CNVcluster'], index=self.CNV_TUMOR.obs.index)
        df_tumor['CNVcluster'] = 'T'+df_tumor['CNVcluster']

        return df_normal, df_tumor


def filter_genes(A, B, min_cells):
    cellsA = np.sum(A.X > 0, axis=0)  # cells expressing the gene
    cellsB = np.sum(B.X > 0, axis=0)  # cells expressing the gene

    cellsA = np.array(cellsA).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array
    cellsB = np.array(cellsB).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array

    ix = np.logical_and(cellsA > min_cells,
                        cellsB > min_cells)

    return A.var.iloc[ix].index


def denoising(Qnormal, Qtumor, sd_amp=1.5):
    """
    for each locus (var), look at the spread in the normal group.
    Anything within (sd_amp * STD) is set to 0
    """
    mean_ref_vals = np.mean(Qnormal.X)
    mean_ref_sd = np.std(Qnormal.X) * sd_amp

    upper_bound = mean_ref_vals + mean_ref_sd
    lower_bound = mean_ref_vals - mean_ref_sd

    mask = np.logical_and(Qtumor.X > lower_bound, Qtumor.X < upper_bound)
    Qtumor.X[mask] = mean_ref_vals

    mask = np.logical_and(Qnormal.X > lower_bound, Qnormal.X < upper_bound)
    Qnormal.X[mask] = mean_ref_vals

    return Qnormal, Qtumor

def get_pyramid_weighting(gene_window):
    ramp = np.linspace(0, 1, gene_window//2).tolist()
    # actually, should be this, the above has a flat part in the center
    # ramp = np.linspace(0, 1, 1+gene_window//2).tolist()[:-1]

    pyramid_weighting = ramp + [1] + ramp[::-1]
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
    center_gene = []
    # for fast lookup, generate a table of genename -> index(int) in adata.X[,]
    # such that adata[, gene].X == adata.X[:, gene_index[gene]]
    gene_index = {gene: ix for ix, gene in enumerate(adata.var.index)}
    X = adata.X  # this causes alot of overhead (its not a simple attribute lokup when adata is a VIEW!!). Hence do it outside the loop
    for i in range(0, len(q)-gene_window, offset):
        genes = q.iloc[i:(i+gene_window)].index

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

    if len(smoothed) > 1:
        smoothed = np.stack(smoothed, axis=1)
    else:
        smoothed = None

    pos = pd.DataFrame(pos, columns=['start'])
    pos['chromosome_name'] = chromosome
    pos['end'] = pos['start'] + gene_window
    pos['middle'] = pos['start'] + (gene_window - 1) // 2
    pos['middle_gene'] = center_gene

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


def split_tumor_normal(adata, ref_field, ref_groups):

    Qnormal = adata[adata.obs[ref_field].isin(ref_groups)]
    Qtumor = adata[~adata.obs[ref_field].isin(ref_groups)]
    return Qnormal, Qtumor


def _preprocess(Qnormal, Qtumor):
    """
    - splitting into reference samples and tumor samples
    - centering on the reference
    - clipping
    """

    gene_ix = filter_genes(Qnormal, Qtumor, 10)
    Qnormal = Qnormal[:, gene_ix]
    Qtumor = Qtumor[:, gene_ix]

    # center based on the reference, before smoothing!
    if issparse(Qnormal.X) or issparse(Qtumor.X):
        print('loading full X.A!!')
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

    Qnormal, Qtumor = split_tumor_normal(adata, ref_field, ref_groups)

    Qnormal, Qtumor = _preprocess(Qnormal, Qtumor)
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


def plotting(S: AnnData, row_color_fields, clustering=None, figsize=(20, 20), vmin=0.5, vmax=1.5):

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
        colormaps[f] = colormap
        # celltype_colors = [colormap[ct] for ct in S.obs[row_color_field]]
    color_df = pd.DataFrame(color_df).T

    if clustering is None:
        print('Clustering')
        clustering = fastcluster.linkage(S.X, method='ward')

    print('Drawing')
    X = pd.DataFrame(S.X, index=S.obs.index)
    g = sns.clustermap(X, col_cluster=False, cmap="bwr",
                       vmin=vmin, vmax=vmax,
                       row_linkage=clustering,
                       col_colors=chrom_colors,
                       row_colors=color_df,
                       xticklabels=False,
                       yticklabels=False,
                       figsize=figsize
                       )

    leg = []
    for f in row_color_fields:
        legend_TN = [mpatches.Patch(color=color, label=ct) for ct, color in colormaps[f].items()]
        leg.extend(legend_TN)

    l2 = g.ax_heatmap.legend(loc='center left', bbox_to_anchor=(1.01, 0.85),
                             handles=leg, frameon=True)
    l2.set_title(title=f, prop={'size': 10})

    return g
