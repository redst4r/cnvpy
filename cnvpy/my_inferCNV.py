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
        self.gene_var = None  # the adata.var from the original data; used to look up genomic coordinates

    def infer(self, adata, ref_field, ref_groups):
        """
        running the inferCNV method, basically smoothing along the chromosomes
        """
        self.gene_var = adata.var.copy()
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

    def cut_tree(self, n_clusters, which, key='cnv_cluster'):

        clusters = cut_tree(self.linkage_tumor, n_clusters=n_clusters).flatten()
        cnv_cluster_df = pd.DataFrame(
            clusters,
            index=self.CNV_TUMOR.obs.index if which == 'tumor' else self.CNV_NORMAL.obs.index,
            columns=[key]
        )
        if which == 'tumor':
            # if the label is already there, drop it and add the new clustering
            if key in self.CNV_TUMOR.obs.columns:
                self.CNV_TUMOR.obs.drop(key, axis=1, inplace=True)
            self.CNV_TUMOR.obs = self.CNV_TUMOR.obs.merge(cnv_cluster_df, left_index=True, right_index=True, how='left')
        else:
            # if the label is already there, drop it and add the new clustering
            if key in self.CNV_NORMAL.obs.columns:
                self.CNV_NORMAL.obs.drop(key, axis=1, inplace=True)
            self.CNV_NORMAL.obs = self.CNV_NORMAL.obs.merge(cnv_cluster_df, left_index=True, right_index=True, how='left')

    def plotting(self, row_color_fields, which, vmin=0.5, vmax=1.5, figsize=(20, 20), colormaps_row=None, interesting_genes=None):
        """
        plot the heatmap of the CNV profiles of either N or Tv
        """
        assert which in ['normal', 'tumor']
        assert self.linkage_normal is not None and self.linkage_tumor is not None, "not clustered yet, run .cluster()"
        S = self.CNV_NORMAL if which == 'normal' else self.CNV_TUMOR
        clustering = self.linkage_normal if which == 'normal' else self.linkage_tumor

        g = plotting(S, row_color_fields, clustering=clustering, vmin=vmin, vmax=vmax, figsize=figsize, colormaps_row=colormaps_row)
        if interesting_genes:
            genes_of_interest_dict = get_gene_coords(self, interesting_genes)
            for gene, index in genes_of_interest_dict.items():
                g.ax_heatmap.vlines(index, 0, len(S))
                g.ax_heatmap.text(index, 0, s=gene, fontdict={'size':20}, rotation=90)
        return g

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


def get_gene_coords(CNV, genenames):
    """
    locate a gene in the columns of the CNV heatmap
    """
    # get all gene-names of the windows
    genemap = pd.concat([
        pd.DataFrame([{
                'pos': i,
                'genes': genes.values,
                'chromosome': chrom
                } for i, genes, _,  _ in chrom_window_generator(CNV.gene_var, chrom)
            ]) for chrom in CHROMOSOMES
        ])

    gene_locations = {}
    for g in genenames:
        _df = genemap[genemap['genes'].apply(lambda x: g in x)]

        # in each window find the distance of the query to the center of teh window we want the window where the gene is closest to center
        _df['distance'] = _df.genes.apply(lambda x: np.abs(len(x)//2-list(x).index(g)) if g in x else np.inf)
        the_row = _df.sort_values('distance').iloc[0]
        pos = the_row['pos']
        chrom = the_row['chromosome']

        ix = np.logical_and(CNV.CNV_TUMOR.var.chromosome_name == chrom,
                            CNV.CNV_TUMOR.var.start == pos)
        gene_locations[g] = np.where(ix)[0][0]

    return gene_locations

# def get_gene_ix_inCNVmap(CNV, genename):
#     start, end, chrom = CNV.gene_var.loc[genename][['start_position', 'end_position', 'chromosome_name']]
#     q = CNV.CNV_TUMOR.var.query('chromosome_name==@chrom and @start>=genomic_start and @end<=genomic_end')
#     q['distance_to_start'] = q['genomic_start'] - start
#     q['distance_to_end'] = q['genomic_end'] - end
#     q['distance'] = q['distance_to_start'] + q['distance_to_end']
#     return q


def filter_genes(A, B, min_cells):
    cellsA = np.sum(A.X > 0, axis=0)  # cells expressing the gene
    cellsB = np.sum(B.X > 0, axis=0)  # cells expressing the gene

    cellsA = np.array(cellsA).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array
    cellsB = np.array(cellsB).flatten()  # in case A.X is a sparse matrix, we wnat a 1D array

    ix = np.logical_and(cellsA > min_cells,
                        cellsB > min_cells)

    return A.var.iloc[ix].index


def denoising(CNV, sd_amp=1.5):
    """
    Denoising the CNV profile.
    for each locus (var), look at the spread in the normal group.
    Anything within (sd_amp * STD) is set to 0.

    This creates a new inferCNV object with the denoised profile.
    The original clustering is preserved though!
    """
    mean_ref_vals = np.mean(CNV.CNV_NORMAL.X)
    mean_ref_sd = np.std(CNV.CNV_NORMAL.X) * sd_amp

    upper_bound = mean_ref_vals + mean_ref_sd
    lower_bound = mean_ref_vals - mean_ref_sd

    denoisedCNV = inferCNV()

    mask = np.logical_and(CNV.CNV_TUMOR.X > lower_bound, CNV.CNV_TUMOR.X < upper_bound)
    denoisedCNV.CNV_TUMOR = CNV.CNV_TUMOR.copy()
    denoisedCNV.CNV_TUMOR.X[mask] = mean_ref_vals

    mask = np.logical_and(CNV.CNV_NORMAL.X > lower_bound, CNV.CNV_NORMAL.X < upper_bound)
    denoisedCNV.CNV_NORMAL = CNV.CNV_NORMAL.copy()
    denoisedCNV.CNV_NORMAL.X[mask] = mean_ref_vals

    denoisedCNV.linkage_normal = CNV.linkage_normal.copy()
    denoisedCNV.linkage_tumor = CNV.linkage_tumor.copy()

    denoisedCNV.gene_var = CNV.gene_var.copy()
    return denoisedCNV


def get_pyramid_weighting(gene_window):
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


def is_categorical(array_like):
    return array_like.dtype.name == 'category'


def plotting(S: AnnData, row_color_fields, clustering=None, figsize=(20, 20), vmin=0.5, vmax=1.5, plot_dendrogram=True, colormaps_row=None):

    if not isinstance(row_color_fields, list):
        row_color_fields = [row_color_fields]

    chrom_colormap = {c: godsnot_64[i] for i, c in enumerate(CHROMOSOMES)}
    chrom_colors = [chrom_colormap[i] for i in S.var.chromosome_name]

    color_df = []
    colormaps = {}
    for f in row_color_fields:
        if is_categorical(S.obs[f]):
            types = S.obs[f].cat.categories
        else:
            types = sorted(S.obs[f].unique())

        if colormaps_row is None or f not in colormaps_row:
            cmap = godsnot_64
            colormap = {ct: cmap[i] for i, ct in enumerate(types)}
        else:
            cmap = colormaps_row[f]
            colormap = {ct: cmap[ct] for ct in types}
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
    if not plot_dendrogram:
        g.ax_row_dendrogram.set_visible(False)

    leg = []
    for f in row_color_fields:
        legend_TN = [mpatches.Patch(color=color, label=ct) for ct, color in colormaps[f].items()]
        leg.extend(legend_TN)

    l2 = g.ax_heatmap.legend(loc='center left', bbox_to_anchor=(1.01, 0.85),
                             handles=leg, frameon=True)
    l2.set_title(title=f, prop={'size': 10})

    return g
