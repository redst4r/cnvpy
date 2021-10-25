import pandas as pd
from anndata import AnnData
from sctools.scplotting import godsnot_64
import fastcluster
import seaborn as sns
import matplotlib.patches as mpatches
import itertools
from cnvpy.utils import CHROMOSOMES


def is_categorical(array_like):
    return array_like.dtype.name == 'category'


def plotting(S: AnnData, row_color_fields, clustering=None, figsize=(20, 20), vmin=0.5, vmax=1.5, plot_dendrogram=True, colormaps_row=None):

    if not isinstance(row_color_fields, list):
        row_color_fields = [row_color_fields]

    bw_map = itertools.cycle(['black', 'grey'])
    # chrom_colormap = {c: godsnot_64[i] for i, c in enumerate(CHROMOSOMES)}
    chrom_colormap = {c: color for c, color in zip(CHROMOSOMES, bw_map)}
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


#
# import plotly.graph_objects as go
# df_reordered = df_X.copy()
#
# ix_row = leaves_list(Z_row)
# ix_col = leaves_list(Z_col)
# df_reordered = df_reordered.iloc[ix_row]
# df_reordered = df_reordered.iloc[:, ix_col]
#
# fig = go.Figure(data=go.Heatmap(
#                    z=df_reordered.T,
#                    x=df_reordered.index.tolist(),
#                    y=['L'+ _ for _ in df_reordered.columns]
# ),
# )
# fig.update_layout(width=3000, height=1000)
# fig.show()





# diagnosis_cmap = {
#     '?': 'grey',  # ?
#     'D': 'orange', # D
#     'DT': 'grey', # DT
#     'M':'#FCE45B', #M
#     'MDT': 'grey', # mix
#     'MT': 'grey', # mix
#     'T': 'red', #T
#     'TDM':'grey', # mix
#     'N': 'darkgreen',
#     'N(stomach)': 'green'
# }

diagnosis_cmap = {
    '?': 'grey',  # ?
    'D': 'blue',  # D
    'DT': 'grey',  # DT
    'M': '#FCE45B',  # M
    'MDT': 'grey',  # mix
    'MT': 'grey',  # mix
    'T': 'red',  # T
    'TDM': 'grey',  # mix
    'N': 'darkgreen',
    'NE': 'darkgreen',
    'N(stomach)': 'green',
    'NS': 'green'
}
is_tumor_cmap = {
    True: 'red',
    False: 'white'
}
is_dysplasia_cmap = {
    True: 'orange',
    False: 'white'
}
is_metaplasia_cmap = {
    True: '#FCE45B',
    False: 'white'
}
