import os
import numpy as np
import pandas as pd
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from GeneAttribution import ComparisonAnalysis, sampling

        
def onegroup_diff(data, label,i):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[1]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")
        print(gpu)
    subtype_label = label[label.cluster==i]
    subtype_data = data.loc[subtype_label.index]
    CA = ComparisonAnalysis(subtype_data, subtype_label, dtype='TPM')
    marker_res = CA.find_marker('cluster', threshold=0.05, type='difference')
        
    out_path = './data/output/attrs/Breast/subtype'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    marker_res.to_csv(
        os.path.join(out_path, 'cluster_{}_res.tsv'.format(i)),
            sep='\t'
    )
        
    for k, v in CA._marker_attrs_.items():
        v.to_csv(
            os.path.join(out_path, 'cluster_{}_attrs.tsv'.format(i)),
            sep='\t')

    
    for k, v in CA.marker_cofreq.items():
        v.to_csv(
            os.path.join(out_path, 'cluster_{}_cofreq.tsv'.format(i)),
            sep='\t')
    
    for k, v in CA.marker_occurences.items():
        v.to_csv(
            os.path.join(out_path, 'cluster_{}_occurences.tsv'.format(i)),
            sep='\t')
        

def multiprocessing_all_diff():
    cluster_label = pd.read_csv('./attr_src/cluster_label.tsv',
                            index_col=0, sep='\t')
    meta = pd.read_csv('./data/TCGA_GTEx/meta_info.tsv',
                    sep= '\t', index_col=0)
    cluster_label.index = meta.loc[cluster_label.index, 'entity_submitter_id'] 
    cluster_label['cluster'] = cluster_label['raw_data_0.2_KMeans']
    data = pd.read_csv('./data/Breast/breast_data.tsv',
                        index_col=0, sep='\t')

    import multiprocessing
    p0 = multiprocessing.Process(target=onegroup_diff, 
                                 args=(data, cluster_label, 
                                       0)
                                 )
    p1 = multiprocessing.Process(target=onegroup_diff, 
                                 args=(data, cluster_label, 
                                       1)
                                 )
    p2 = multiprocessing.Process(target=onegroup_diff, 
                                 args=(data, cluster_label, 
                                       2)
                                 )
    p3 = multiprocessing.Process(target=onegroup_diff, 
                                 args=(data, cluster_label, 
                                       3)
                                 )
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    
    p4 = multiprocessing.Process(target=onegroup_diff, 
                                 args=(data, cluster_label, 
                                       4)
                                 )
    p4.start()
    p4.join()

    
multiprocessing_all_diff()
