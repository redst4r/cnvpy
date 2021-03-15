import pandas as pd
import os
import tempfile
from scipy.sparse import issparse


def adata2inferCNV(adata, annotation_field, folder):
    """
    turns adata into the format used by inferCNV. Puts all files in a
    user-specified folder (data.txt, annotation.txt)
    """
    os.mkdir(folder)
#     tempdir = tempfile.mkdtemp()
    tempdir = folder
    df = pd.DataFrame(adata.X.A) if issparse(adata.X) else pd.DataFrame(adata.X)
    df.index = adata.obs.index
    df.columns = adata.var.index
    print('writing count matrix')
    df.T.to_csv(f'{tempdir}/data.txt', sep='\t')

    df_annotation = adata.obs[[annotation_field]]
    print('writing annotation')
    df_annotation.to_csv(f'{tempdir}/annotation.txt', header=False, sep='\t')
    print(df_annotation[annotation_field].unique())

    return tempdir


def create_docker_call(tempdir, reference_group_name):
    """
    docker call for inferCNV

    TODO currently doesnt work, the script /infercnv/scripts/inferCNV.R fails
    inside the container
    """

    gene_order_file = '/home/michi/postdoc_seattle/CRUK/CNV/gencode_v19_gene_pos.txt'
    # need to copy this into the tempdir
    os.system(f'cp {gene_order_file} {tempdir}')

    data_file = 'data.txt'
    annotation_file = 'annotation.txt'
    cutoff = 0.1
    outdir = 'output'
    s = \
    f"""
    docker run -v {tempdir}:/data -w /data --rm -it trinityctat/infercnv /infercnv/scripts/inferCNV.R \
         --raw_counts_matrix="{data_file}" \
         --annotations_file="{annotation_file}" \
         --gene_order_file="gencode_v19_gene_pos.txt" \
         --ref_group_names="{reference_group_name}" \
         --cutoff={cutoff} \
         --out_dir="{outdir}" \
         --cluster_by_groups \
         --denoise \
         --plot_steps
    """
    return s
