from rnaseqtools.biomart_mapping import biomart_query_all
import numpy as np


def get_genes_in_genomic_region(adata, chrom, pos_start, pos_end):
    genes = adata.var.query(
        'chromosome_name == @chrom and start_position >= @pos_start and start_position <= @pos_end'
    )
    return list(genes.index.values)


def annotate_genomic_coordinates(adata, join_on_symbol=False):
    df_biomart = biomart_query_all()
    df_biomart['chromosome_name'] = df_biomart['chromosome_name'].apply(lambda x: str(x))

    chroms = [str(_) for _ in range(1, 23)] + ['X', 'Y', 'MT']
    df_biomart = df_biomart[df_biomart.chromosome_name.isin(chroms)]

    if join_on_symbol:
        _tmp = df_biomart[['hgnc_symbol', 'chromosome_name', 'start_position', 'end_position']].drop_duplicates()

        var = adata.var.reset_index().merge(_tmp, left_on='index', right_on='hgnc_symbol', how='left').set_index('index')
        var = var.drop_duplicates('hgnc_symbol')  # sometimes theres some odd duplicate genenames with slightly different coords
    else:
        _tmp = df_biomart[['ensembl_gene_id', 'chromosome_name', 'start_position', 'end_position']].drop_duplicates()
        indexname = adata.var.index.name
        var = adata.var.reset_index().merge(_tmp, left_on='gene_ids', right_on='ensembl_gene_id', how='left').set_index(indexname)

    # in case we couldnt find info
    # this should only happen in the join_symbol case!
    shared_genes = [_ for _ in adata.var.index if _ in var.index]
    adata = adata[:, shared_genes]

    adata.var = var
    adata.var.chromosome_name = adata.var.chromosome_name.replace({np.nan: 'NA'})
    return adata
