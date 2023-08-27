import pandas as pd
import tensorflow as tf

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import umap_visualization, log2_feature

#TCGA and GTEx UMAP visulization
label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                    index_col=0, sep='\t')
data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv',
                   index_col=0, sep='\t')
zc = pd.read_csv('./data/output/zc.tsv',
                 index_col=0, sep='\t')
zb = pd.read_csv('./data/output/zb.tsv',
                 index_col=0, sep='\t')
z_concat = pd.read_csv('./data/output/z_concat.tsv',
                       index_col=0, sep='\t')
bfx = pd.read_csv('./data/output/bfx.tsv',
                  index_col=0, sep='\t')
bfx = log2_feature(bfx, dtype='TPM')
_, embed_x = umap_visualization(data, label, ['batch'], z_dim=100,
                                n_neighbors=25, min_dist=0.5)
_, embed_zc = umap_visualization(zc, label, ['batch'], z_dim=100,
                                 n_neighbors=25, min_dist=0.5)
_, embed_zb = umap_visualization(zb, label, ['batch'], z_dim=100,
                                 n_neighbors=25, min_dist=0.5)
_, embed_zconcat = umap_visualization(z_concat, label, ['batch'], z_dim=100,
                                      n_neighbors=25, min_dist=0.5)
_, embed_bfx = umap_visualization(bfx, label, ['batch'], z_dim=100,
                                  n_neighbors=25, min_dist=0.5)
embed_x.to_csv('./figure/UMAP/x.tsv', sep='\t')
embed_zc.to_csv('./figure/UMAP/zc.tsv', sep='\t')
embed_zb.to_csv('./figure/UMAP/zb.tsv', sep='\t')
embed_zconcat.to_csv('./figure/UMAP/z_concat.tsv', sep='\t')
embed_bfx.to_csv('./figure/UMAP/bfx.tsv', sep='\t')

#all dataset umap for specific tissue
all_dataset_label = pd.read_csv('./data/output/all_dataset_label.tsv',
                                index_col=0, sep='\t')
all_dataset_x = pd.read_csv('./data/output/all_dataset_log2_x.tsv',
                            index_col=0, sep='\t')
all_dataset_zc = pd.read_csv('./data/output/all_dataset_zc.tsv',
                             index_col=0, sep='\t')
all_dataset_bfx = pd.read_csv('./data/output/all_dataset_bfx.tsv',
                             index_col=0, sep='\t')
all_dataset_bfx = log2_feature(all_dataset_bfx, dtype='TPM')

embed_x, embed_zc, embed_bfx = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
ps = ['Hematopoietic and reticuloendothelial systems',
      'Bronchus and lung', 'Kidney']
for i in ps:
    label_tmp = all_dataset_label[all_dataset_label['primary_site']==i]
    x_tmp = all_dataset_x.loc[label_tmp.index]
    zc_tmp = all_dataset_zc.loc[label_tmp.index]
    bfx_tmp = all_dataset_bfx.loc[label_tmp.index]
    _, embed_x_ = umap_visualization(x_tmp, label_tmp,
                                    ['batch'], z_dim=100,
                                    n_neighbors=25, min_dist=0.5)
    _, embed_zc_ = umap_visualization(zc_tmp, label_tmp,
                                    ['batch'], z_dim=100,
                                    n_neighbors=25, min_dist=0.5)
    _, embed_bfx_ = umap_visualization(bfx_tmp, label_tmp,
                                    ['batch'], z_dim=100,
                                    n_neighbors=25, min_dist=0.5)
    embed_x = pd.concat([embed_x, embed_x_])
    embed_zc = pd.concat([embed_zc, embed_zc_])
    embed_bfx = pd.concat([embed_bfx, embed_bfx_])
embed_x.to_csv('./figure/UMAP/all_dataset_x.tsv', sep='\t')
embed_zc.to_csv('./figure/UMAP/all_dataset_zc.tsv', sep='\t')
embed_bfx.to_csv('./figure/UMAP/all_dataset_bfx.tsv', sep='\t')

#breast data umap
breast_label = pd.read_csv('./data/Breast/breast_label.tsv',
                           index_col=0, sep='\t')
breast_data = pd.read_csv('./data/Breast/breast_data.tsv',
                            index_col=0, sep='\t')
breast_data = log2_feature(breast_data, dtype='TPM')
breast_zc = pd.read_csv('./data/output/breast_zc.tsv',
                        index_col=0, sep='\t')
breast_zb = pd.read_csv('./data/output/breast_zb.tsv',
                        index_col=0, sep='\t')
breast_bfx = pd.read_csv('./data/output/breast_bfx.tsv',
                         index_col=0, sep='\t')
breast_bfx = log2_feature(breast_bfx, dtype='TPM')
breast_zc = breast_zc.loc[breast_label.index]
breast_zb = breast_zb.loc[breast_label.index]
breast_data = breast_data.loc[breast_label.index]
breast_bfx = breast_bfx.loc[breast_label.index]
_, embed_x = umap_visualization(breast_data, breast_label,
                                ['batch'], z_dim=100,
                                n_neighbors=200, min_dist=0.99)
_, embed_zc = umap_visualization(breast_zc, breast_label,
                                 ['batch'], z_dim=100,
                                 n_neighbors=8, min_dist=0.99)
_, embed_zb = umap_visualization(breast_zb, breast_label,
                                 ['batch'], z_dim=100,
                                 n_neighbors=8, min_dist=0.99)
_, embed_bfx = umap_visualization(breast_bfx, breast_label,
                                  ['batch'], z_dim=100,
                                  n_neighbors=200, min_dist=0.99)
embed_x.to_csv('./figure/UMAP/breast_x.tsv', sep='\t')
embed_zc.to_csv('./figure/UMAP/breast_zc.tsv', sep='\t')
embed_zb.to_csv('./figure/UMAP/breast_zb.tsv', sep='\t')
embed_bfx.to_csv('./figure/UMAP/breast_bfx.tsv', sep='\t')

