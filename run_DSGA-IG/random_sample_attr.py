import os
import pandas as pd
import numpy as np
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import Wasserstein_Distance, gene2img, load_model
from GeneAttribution import ComparisonAnalysis

def time_cost(data, label):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[1]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")
        print(gpu)
    
    import time
    
    t1 = time.time()
    CA = ComparisonAnalysis(data, label, dtype='log2TPM')
    res = CA.find_marker('cancer_type', threshold=0.05)
    time_cost = time.time() - t1
    return time_cost


def farthest_point_sample(data, npoint, dtype):
    model = load_model(ckpt='latest', ckpt_path='', z_dim=100, only_encoder=True)

    seeds = np.zeros(npoint)
    distance = np.ones(data.shape[0]) * 1e10
    
    import random
    random.seed(10)
    seed = random.randint(0, data.shape[0])
    
    for i in range(npoint):
        seeds[i] = seed
        d = one2other_distance(model, data, seed, dtype)
        mask = d < distance
        distance[mask] = d[mask]

        seed = np.argmax(distance)

    sample_data = data.iloc[seeds]
    return sample_data


def one2other_distance(model, data, idx, dtype):
    data = gene2img(data, dtype=dtype)
    
    means, logvars = model(data)
    
    d = Wasserstein_Distance(means, logvars, means[idx], logvars[idx])
    return d
    
    
def sampling_cancer_diff(data, label, primary_site, percent, gpu_num=0):

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[gpu_num]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")
        print(gpu)
                
        out_path = os.path.join('./data/output/attrs/point_cloud_sampling',
                                primary_site)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        CA = ComparisonAnalysis(data, label, dtype='log2TPM')
        diff_res = CA.find_difference('cancer_type', threshold=0.05)
        
        diff_res.to_csv(
            os.path.join(out_path, '{}_diff_res.tsv'.format(percent)),
                sep='\t'
        )
        
        for k, v in CA._diff_attrs_.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_attrs.tsv'.format(percent)),
                sep='\t')

        
        for k, v in CA.diff_cofreq.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_cofreq.tsv'.format(percent)),
                sep='\t')
        
        for k, v in CA.diff_occurences.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_occurences.tsv'.format(percent)),
                sep='\t')

data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv', index_col=0,
                   sep='\t')
label = pd.read_csv('./data/TCGA_GTEx/y_data.tsv', index_col=0,
                    sep='\t')

primary_site = ['Testis']
percent = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
gpu_num = 0
        
import multiprocessing
    
for ps in primary_site:
    tmp_label = label[label.primary_site == ps]
    tmp_label['cancer_type'][tmp_label['cancer_type'] != 'Normal'] = 'Tumor'
    tmp_data = data.loc[tmp_label.index]
        
    for p in percent:
        
        npoint = round(len(sample_data) * p)
        sample_data = farthest_point_sample(tmp_data, npoint, 'log2TPM')
        sample_label = tmp_label.loc[sample_data.index]

        process = multiprocessing.Process(target=sampling_cancer_diff,
                                            args=(sample_data, sample_label,
                                                    ps, p, gpu_num))        
        process.start()
        if gpu_num == 0 :
            gpu_num = 1
        else:
            gpu_num = 0

    
    
