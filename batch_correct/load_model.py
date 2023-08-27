import pandas as pd
import tensorflow as tf

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import generate_data, generate_batch_data, reverse_tpm_feature, log2_feature, generate_distribution

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu1 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu1, True)
        tf.config.experimental.set_visible_devices([gpu1],"GPU")
        print(gpu1)    
        
    #load TCGA and GTEx data
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')
    data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv',
                       index_col=0, sep='\t')
    x = reverse_tpm_feature(data)
    zc, bfx = generate_data(data, dtype='log2TPM', outdtype='TPM')
    zb = generate_batch_data(data, dtype='log2TPM')
    mean, logvar = generate_distribution(data, dtype='log2TPM')
    z_concat = pd.concat([zb, zc], axis=1)
    x.to_csv('./data/output/x.tsv', sep='\t')
    zc.to_csv('./data/output/zc.tsv', sep='\t')
    zb.to_csv('./data/output/zb.tsv', sep='\t')
    z_concat.to_csv('./data/output/z_concat.tsv', sep='\t')
    bfx.to_csv('./data/output/bfx.tsv', sep='\t')
    mean.to_csv('./data/output/mean.tsv', sep='\t')
    logvar.to_csv('./data/output/logvar.tsv', sep='\t')
    
    #load all dataset
    CPTAC_label = pd.read_csv('./data/CPTAC/cptac_label.tsv',
                              index_col=0, sep='\t')
    CPTAC_data = pd.read_csv('./data/CPTAC/cptac_fpkm.tsv',
                             index_col=0, sep='\t')
    TARGET_label = pd.read_csv('./data/TARGET/target_label.tsv',
                               index_col=0, sep='\t')
    TARGET_data = pd.read_csv('./data/TARGET/target_fpkm.tsv',
                              index_col=0, sep='\t')
    TCGA_nonus_label = pd.read_csv('./data/TCGA_non-us/tcga_non-us_label.tsv',
                                   index_col=0, sep='\t') 
    TCGA_nonus_data = pd.read_csv('./data/TCGA_non-us/tcga_non-us_fpkm_uq.tsv',
                                  index_col=0, sep='\t')
    other_label = pd.concat([CPTAC_label, TARGET_label, TCGA_nonus_label])
    other_data = pd.concat([CPTAC_data, TARGET_data, TCGA_nonus_data])
    other_data = other_data[data.columns]
    other_data.fillna(0, inplace=True)
    other_data[other_data<0] = 0
    other_data = log2_feature(other_data, dtype='FPKM')
    
    label = pd.concat([label, other_label])
    data = pd.concat([data, other_data])
    zc, bfx = generate_data(data, dtype='log2TPM', outdtype='TPM')
    data.to_csv('./data/output/all_dataset_log2_x.tsv', sep='\t')
    zc.to_csv('./data/output/all_dataset_zc.tsv', sep='\t')
    bfx.to_csv('./data/output/all_dataset_bfx.tsv', sep='\t')
    label.to_csv('./data/output/all_dataset_label.tsv', sep='\t')
    
    #load breast data
    breast_label = pd.read_csv('./data/Breast/breast_label.tsv',
                               index_col=0, sep='\t')
    breast_data = pd.read_csv('./data/Breast/breast_data.tsv',
                              index_col=0, sep='\t')
    breast_zc, breast_bfx = generate_data(breast_data, dtype='TPM', outdtype='TPM')
    breast_zb = generate_batch_data(breast_data, dtype='TPM')
    mean, logvar = generate_distribution(breast_data, dtype='TPM')
    breast_zc.to_csv('./data/output/breast_zc.tsv', sep='\t')
    breast_zb.to_csv('./data/output/breast_zb.tsv', sep='\t')
    breast_bfx.to_csv('./data/output/breast_bfx.tsv', sep='\t')
    mean.to_csv('./data/output/breast_mean.tsv', sep='\t')
    logvar.to_csv('./data/output/breast_logvar.tsv', sep='\t')
    
    
    