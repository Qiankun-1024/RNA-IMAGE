import pandas as pd

counts = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/counts_features.tsv',
                     sep = '\t', index_col=0)
breast = pd.read_csv('/vol1/cuipeng_group/qiankun/multilabel_classification/GTEx/gtex_breast_counts.tsv',
                     sep = '\t', index_col=0)
counts = pd.concat([counts, breast])
label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv', sep = '\t', index_col=0)
data = pd.read_csv('/vol2/cuipeng_group/qiankun/RNA-IMAGE/data/TCGA_GTEx/log2_x_data.tsv', sep = '\t', index_col=0)
counts = counts.loc[label.index, data.columns]
counts = counts.fillna(0)
counts.to_csv('/vol2/cuipeng_group/qiankun/RNA-IMAGE/data/TCGA_GTEx/counts_data.tsv',sep='\t')
