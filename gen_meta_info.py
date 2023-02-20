import numpy as np
import pandas as pd

y = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/TCGA_GTEx/y_data.tsv', index_col=0, sep='\t')
meta = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/meta_info.tsv', index_col=0, sep='\t')
meta = meta[['disease_type', 'Sample Type', 'tumor_stage']]
label = pd.concat([y, meta], axis=1, join='outer')
label = label.loc[y.index]
label = label.fillna('Normal')
label.replace(['Stage IA', 'Stage IB'], 'Stage I', inplace=True)
label.replace(['Stage IIA', 'Stage IIB', 'Stage IIC'], 'Stage II', inplace=True)
label.replace(['Stage IIIA', 'Stage IIIB', 'Stage IIIC'], 'Stage III', inplace=True)
label.replace(['Stage IVA', 'Stage IVB', 'Stage IVC'], 'Stage IV', inplace=True)
label.replace(['Stage IS', 'Stage X'], 'None', inplace=True)

label.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/TCGA_GTEx/all_label.tsv', sep='\t')