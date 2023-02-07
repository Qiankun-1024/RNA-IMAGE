import os
import pandas as pd
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from GeneAttribution import ComparisonAnalysis, sampling


    
def all_cancer_type_diff(data, label, cancer_type, primary_site):
    
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[1]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")
        print(gpu)
    
    for ct, ps in zip(cancer_type, primary_site):
        tmp_label = label[(label['cancer_type'].isin([ct, 'Normal'])) & (label['primary_site']==ps)]
        tmp_data = data.loc[tmp_label.index]
        
        sample_data, sample_label = \
            sampling.__dict__['cloud_point_sample'](tmp_data, tmp_label,
                                                    0.25, 'log2TPM')
        
        CA = ComparisonAnalysis(sample_data, sample_label, dtype='log2TPM')
        diff_res = CA.find_difference('cancer_type', threshold=0.05)
        
        out_path = os.path.join('./data/output/attrs/all/sampling_diff', ct)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        diff_res.to_csv(
            os.path.join(out_path, '{}_diff_res.tsv'.format(ct)),
                sep='\t'
        )
        
        for k, v in CA._diff_attrs_.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_attrs.tsv'.format(ct)),
                sep='\t')

        
        for k, v in CA.diff_cofreq.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_cofreq.tsv'.format(ct)),
                sep='\t')
        
        for k, v in CA.diff_occurences.items():
            v.to_csv(
                os.path.join(out_path, '{}_diff_occurences.tsv'.format(ct)),
                sep='\t')


def multiprocessing_all_diff():
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')
    data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv',
                       index_col=0, sep='\t')
    
    cancer_type = ['KICH','PCPG', 'PAAD', 'UCEC', 'MESO', 'LUAD', 'GBM',
                   'SKCM', 'BLCA', 'OV', 'STAD', 'THCA', 'COAD','LGG',
                   'KIRP', 'ACC', 'LIHC', 'PRAD', 'LUSC', 'ESCA',
                   'BRCA', 'KIRC', 'TGCT', 'READ', 'CESC']
    primary_site = ['Kidney', 'Adrenal gland', 'Pancreas', 'Corpus uteri',
                    'Heart, mediastinum, and pleura',
                    'Bronchus and lung', 'Brain', 'Skin', 'Bladder',
                    'Ovary', 'Stomach', 'Thyroid gland', 'Colon', 'Brain',
                    'Kidney','Adrenal gland', 
                    'Liver and intrahepatic bile ducts',
                    'Prostate gland', 'Bronchus and lung', 'Esophagus', 
                    'Breast', 'Kidney', 'Testis', 'Colon' , 'Cervix uteri',]
    cancer_type_A = cancer_type[0:7]
    cancer_type_B = cancer_type[7:14]
    cancer_type_C = cancer_type[14:20]
    cancer_type_D = cancer_type[20:25]
    primary_site_A = primary_site[0:7]
    primary_site_B = primary_site[7:14]
    primary_site_C = primary_site[14:20]
    primary_site_D = primary_site[20:25]

    
    import multiprocessing
    process_A = multiprocessing.Process(target=all_cancer_type_diff,
                                        args=(data, label,
                                              cancer_type_A,
                                              primary_site_A))
    process_B = multiprocessing.Process(target=all_cancer_type_diff,
                                        args=(data, label,
                                              cancer_type_B,
                                              primary_site_B))
    process_C = multiprocessing.Process(target=all_cancer_type_diff,
                                        args=(data, label,
                                              cancer_type_C,
                                              primary_site_C))
    process_D = multiprocessing.Process(target=all_cancer_type_diff,
                                        args=(data, label,
                                              cancer_type_D,
                                              primary_site_D))
    
    process_A.start()
    process_B.start()
    process_C.start()
    process_D.start()
        

if __name__ == '__main__':
    multiprocessing_all_diff()

