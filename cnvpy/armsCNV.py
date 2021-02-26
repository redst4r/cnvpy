
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix
from scipy.stats.distributions import chi2
from scipy.stats import norm
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from rnaseqtools.biomart_mapping import biomart_query_all

"""
to determine the CNV on a very coarse, chromosome arm scale

somehow related to this CONICS paper (I forgot)
"""


def likelihood_ratio(llmin, llmax, degrees_of_freedom):
    LR = 2*(llmax-llmin)

    if llmin > llmax:
        print("Warning llmin !< llmax")
        p = 1.0
    else:
        p = chi2.sf(LR, degrees_of_freedom)  # degrees_of_freedom==1 : L2 has 1 DoF more than L1
    return LR, p


def scale(x, axis):
    m = x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    y = (x - m) / s
    return y


def calcNormFactors(adata):
    """
    the mean of each cell

    should be run in log2(CPM/10+1)

    """
    return np.array(adata.X.mean(1)).flatten()


def annotate_genomic_coordinates(adata, join_on_symbol=False):
    df_biomart = biomart_query_all()

    chroms = [str(_) for _ in range(1, 23)] + ['X', 'Y', 'MT']
    df_biomart = df_biomart[df_biomart.chromosome_name.isin(chroms)]
    if join_on_symbol:
        _tmp = df_biomart[['hgnc_symbol', 'chromosome_name', 'start_position', 'end_position']].drop_duplicates()
        var = adata.var.reset_index().merge(_tmp, left_on='index', right_on='hgnc_symbol', how='left').set_index('index')
        var = var.drop_duplicates('hgnc_symbol')  # sometimes theres some odd duplicate genenames with slightly different coords
    else:
        _tmp = df_biomart[['ensembl_gene_id', 'chromosome_name', 'start_position', 'end_position']].drop_duplicates()
        var = adata.var.reset_index().merge(_tmp, left_on='gene_ids', right_on='ensembl_gene_id', how='left').set_index('index')

    # in case we couldnt find info
    # this should only happen in the join_symbol case!
    shared_genes = [_ for _ in adata.var.index if _ in var.index]
    adata = adata[:, shared_genes]

    adata.var = var
    adata.var.chromosome_name = adata.var.chromosome_name.replace({np.nan: 'NA'})
    return adata


def get_genes_in_genomic_region(adata, chrom, pos_start, pos_end):
    genes = adata.var.query('chromosome_name == @chrom and start_position >= @pos_start and start_position <= @pos_end')
    return list(genes.index.values)


def read_chromosome_arms_coords():
    df = pd.read_csv('chromosome_arm_positions_grch38.txt', sep='\t').set_index('Idf')
    df['Chrom'] = df['Chrom'].apply(str)

    dfx = pd.DataFrame([{'Idf': 'Xp', 'Chrom': 'X', 'Start':0, 'End': 60_000_000, 'Length': 60_000_000},
                        {'Idf': 'Xq', 'Chrom': 'X', 'Start':60_000_000, 'End': 156_000_000, 'Length': 96000000},
                        ]).set_index('Idf')
    df = df.append(dfx)
    return df


def plotChrEnichment(adata, genes, normFactor, plotting=False):
    """
    should be run in log2(CPM/10+1)
    """
    # the matrix is cells x genes (the R version is transposed)

    Y = np.array(adata[:, genes].X.mean(1)).flatten() - normFactor
    # Y is a vector: expression of the genomic region in each cell
    Y = scale(Y, axis=0)  # just scale the entire genomic region to 0mean, 1std

    Y = Y.reshape(-1, 1)
    gmm1, gmm2, LR, pval, ll1, ll2, bic1, bic2, prob_vector = fit_mixture_model(Y, plotting)

    return {'LR': LR, 'pval': pval, 'bic1': bic1, 'bic2': bic2}, prob_vector, Y


def counts_to_conics(adata):
    """
    transforms the counts in adata to log2(CPM/10+1)

    thisreturns a new adata
    """
    # TODO LOTS OF MEMORY!!
    CPM = adata.X.A / np.array(adata.X.sum(1)) * 1_000_000
    X = csr_matrix(np.log2(CPM/10 + 1))

    new_adata = sc.AnnData(X, obs=adata.obs.copy(), var=adata.var.copy())
    return new_adata


def viz_mixture_model(gmm):
    plt.figure()
    for i in range(gmm.n_components):
        sns.distplot(norm(gmm.means_[i], np.sqrt(gmm.covariances_[i])).rvs(100), label=f'C{i}')
    plt.legend()


def loglike(gmm, X):
    likes = np.zeros(X.shape)
    for i in range(gmm.n_components):
        p = gmm.weights_[i] * norm(gmm.means_[i], np.sqrt(gmm.covariances_[i])).pdf(X)
        likes = likes+p
    logp = np.log(likes).sum()
    return logp


def fit_mixture_model(X, plotting=False):
    assert X.ndim == 2 and X.shape[1] == 1

    gmm1 = GaussianMixture(n_components=1, init_params='kmeans', n_init=100, max_iter=1000)
    gmm2 = GaussianMixture(n_components=2, init_params='kmeans', n_init=100, max_iter=1000)
    gmm1.fit(X)
    gmm2.fit(X)

    ll1 = gmm1.score(X) * X.shape[0]  # .score() returns the AVERAGE over all data
    ll2 = gmm2.score(X) * X.shape[0]  # however for the LR test we need the ACTUAL logp, hence we undo the avg

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)

    LR, pval = likelihood_ratio(ll1, ll2, degrees_of_freedom=1)

    clusters = gmm2.predict(X)
    prob = gmm2.predict_proba(X)
    # for the two component model, check which component is closest to 0
    # that one will be labeled 0
    ix = np.argmin(np.abs(gmm2.means_))
    if ix == 1:
        # switch the labels
        clusters = np.array([0 if _ == 1 else 1 for _ in clusters])
        prob = prob[:, 0]  # as 1 identifies the base/unaltered class
    else:
        prob = prob[:, 1]

    if plotting:
        plt.figure(figsize=(15, 5))
        plt.subplot(141)
        sns.distplot(X)
        sns.distplot(X[clusters==0], label='c1')
        sns.distplot(X[clusters==1], label='c2')
        plt.legend()
        plt.xlabel('z-scored expression')
        plt.subplot(142)
        plt.bar(range(2), [bic1, bic2])
        plt.ylabel('BIC')
        plt.title(f'LR-pval: {pval:.3e}')
        plt.subplot(143)
        sns.distplot(prob, bins=20, kde=False)
        plt.xlabel('Posterior prob per datapoint')
        plt.subplot(144)
        plt.scatter(range(len(X)), X, s=1, alpha=0.3)

    return gmm1, gmm2, LR, pval, ll1, ll2, bic1, bic2, prob
