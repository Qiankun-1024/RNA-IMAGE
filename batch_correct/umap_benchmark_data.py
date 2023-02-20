import pandas as pd
import tensorflow as tf

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import umap_visualization
        
label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                    index_col=0, sep='\t')
data_path = './data/output/benchmark_batch_correction'
data_list = ['rpca_counts_pca', 'rpca_tpm_pca', 'cca_counts_pca', 'cca_tpm_pca',
                'liger_counts_pca', 'liger_tpm_pca', 'harmony_counts_pca', 'harmony_tpm_pca',
                'ruvr_counts', 'ruvs_counts', 
                'combat_counts', 'combat_tpm','combat_seq_counts',
                'limma_counts', 'limma_tpm']
for i in data_list:
    pca_path = os.path.join(data_path, (i + '.tsv'))
    pca_data = pd.read_csv(pca_path, sep='\t', index_col=0)
    pca_data.index = label.index
    
    _, umap_data = umap_visualization(pca_data, label, ['batch'], z_dim=100,
                                 n_neighbors=25, min_dist=0.5)
    umap_data.to_csv(os.path.join(data_path, (i + '_umap.tsv')), sep='\t')


