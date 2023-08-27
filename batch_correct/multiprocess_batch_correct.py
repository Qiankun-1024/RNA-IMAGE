import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import chisquare


class k_neighbours_chisquare_test():
    '''
    k nearest neighbours rejection rate
    '''
    def __call__(self, data, batch, k):
        self.batch = batch
        # pca = PCA(n_components=10)
        # pc = pca.fit_transform(data)
        pc = data
        
        indices = self.compute_nearest_neighbors(k, pc)
        pi = batch.value_counts()/len(batch)
        rejection_rates = self.compute_rejection_rates(pi, k, indices)
        return rejection_rates
        
    def compute_nearest_neighbors(self, n_neighbors, data):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        return indices
        
    def compute_rejection_rates(self, pi, k, indices):
        f_df = k * pi
        for i in indices:
            fi = self.batch.iloc[i].value_counts().astype(float)
            f_df = pd.concat([f_df, fi], join='outer', axis=1)
        f_df = f_df.fillna(0)
        f_exp = np.expand_dims(f_df.iloc[:,0].values, axis=1)
        fi = f_df.iloc[:,1:].values
        chisq, p_value = chisquare(fi, f_exp)
        rejection_rates = np.sum(np.where(p_value >= 0.05, 0, 1)) / p_value.size
        return rejection_rates
        


def bisection_compute_rejection_rates(data, batch, a, b):
    knt = k_neighbours_chisquare_test()
    c = int((a+b)/2)
    fa = knt(data, batch, a)
    fb = knt(data, batch, b)
    fc = knt(data, batch, c)
    score = fa
    while (abs(b-a)>10) & (abs(fa-fb)>0.01):
        if fc > fa:
            a = c
        else:
            b = c
        score = bisection_compute_rejection_rates(data, batch, a,b)
    return score


class clustering_evaluation():
    def __init__(self, data, label, label_class='primary_site'):
        self.data = data
        label['class'] = label[label_class]
        le = LabelEncoder()
        self.true_labels = le.fit_transform(label[label_class])
        n_clusters = np.size(np.unique(self.true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.data)
        self.pred_labels = kmeans.labels_
        
    def adjusted_rand_index(self):
        score = metrics.adjusted_rand_score(self.true_labels, self.pred_labels)
        return abs(score)
        
    def mutual_info_score(self):
        score = metrics.adjusted_mutual_info_score(self.true_labels, self.pred_labels) 
        return abs(score)
    
    def silhouette_score(self):
        score = metrics.silhouette_score(self.data, self.pred_labels, metric='euclidean')
        return abs(score)
    

def normal_batch_evaluate(data_type, label, ps_list):
   
    data_path = './data/output/benchmark_batch_correction'
    umap_path = os.path.join(data_path, (data_type + '_umap.tsv'))
    umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
    umap_data.index = label.index
    
    ps_col, data_col, ARI_col, MI_col, silhouette_col, kBET_col = [], [], [], [], [], []
    for ps in ps_list:
        label_tmp = label[(label['primary_site']==ps)& \
                            (label['cancer_type']=='Normal')]
        umap_tmp = umap_data.loc[label_tmp.index]
        ce = clustering_evaluation(umap_tmp, label_tmp, label_class='batch')
        if len(label_tmp)>40:
            label_tmp = label_tmp.groupby('batch').sample(20, random_state=42)
            umap_tmp = umap_data.loc[label_tmp.index]
        a = int(len(umap_tmp) * 0.25)
        b = int(len(umap_tmp) * 0.75)
        rejection_rates = bisection_compute_rejection_rates(
            umap_tmp, label_tmp['batch'], a, b)
        kBET_col.append(rejection_rates)
        ARI_col.append(ce.adjusted_rand_index())
        MI_col.append(ce.mutual_info_score())
        silhouette_col.append(ce.silhouette_score())
        ps_col.append(ps)
        data_col.append(data_type)
    
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col,
                                 'silhouette_score':silhouette_col,
                                 'kBETscore':kBET_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/batch_evaluate/normal/' + data_type + '_cluster_score.tsv', sep='\t')


def multiprocessing_normal_evaluate():
    data_list = ['rpca_counts_pca', 'rpca_tpm_pca', 'cca_counts_pca', 'cca_tpm_pca',
                 'liger_counts_pca', 'liger_tpm_pca', 'harmony_counts_pca', 'harmony_tpm_pca',
                 'ruvr_counts', 'ruvs_counts',
                 'combat_counts', 'combat_tpm','combat_seq_counts',
                 'limma_counts', 'limma_tpm',
                 'zc', 'bfx', 'x']
    
    ps_list = ["Bladder", "Breast", "Colon", "Corpus uteri", "Kidney",
               "Liver and intrahepatic bile ducts", "Prostate gland",
               "Stomach", "Thyroid gland"]
    
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')

    import multiprocessing
    process_A = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[0], label, ps_list))
    process_B = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[1], label, ps_list))
    process_C = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[2], label, ps_list))
    process_D = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[3], label, ps_list))
    process_E = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[4], label, ps_list))
    process_F = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[5], label, ps_list))
    process_G = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[6], label, ps_list))
    process_H = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[7], label, ps_list))
    process_I = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[8], label, ps_list))
    process_J = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[9], label, ps_list))
    process_K = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[10], label, ps_list))
    process_L = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[11], label, ps_list))
    process_M = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[12], label, ps_list))
    process_N = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[13], label, ps_list))
    process_O = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[14], label, ps_list))
    process_P = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[15], label, ps_list))
    process_Q = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[16], label, ps_list))
    process_R = multiprocessing.Process(target=normal_batch_evaluate,
                                        args=(data_list[17], label, ps_list))
    
    process_A.start()
    process_B.start()
    process_C.start()
    process_D.start()
    process_E.start()
    process_F.start()
    process_G.start()
    process_H.start()
    process_I.start()
    process_J.start()
    process_K.start()
    process_L.start()
    process_M.start()
    process_N.start()
    process_O.start()
    process_P.start()
    process_Q.start()
    process_R.start()


def ps_evaluate(data_type, label, ps_list):
   
    data_path = './data/output/benchmark_batch_correction'
    umap_path = os.path.join(data_path, (data_type + '_umap.tsv'))
    umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
    umap_data.index = label.index
    
    ps_col, data_col, ARI_col, MI_col, silhouette_col, kBET_col = [], [], [], [], [], []
    for ps in ps_list:
        label_tmp = label[label['primary_site']==ps]
        counts = label_tmp['cancer_type'].value_counts()
        drop = counts[counts <= 10].index.tolist()
        for i in drop:
            label_tmp[label_tmp==i] = np.nan
        label_tmp.dropna(how='any',inplace=True)
        label_tmp.loc[label['cancer_type']!="Normal", "cancer_type"] = "Tumor"
        umap_tmp = umap_data.loc[label_tmp.index]
        ce = clustering_evaluation(umap_tmp, label_tmp, label_class='cancer_type')
        if len(label_tmp)>40:
            label_tmp = label_tmp.groupby('cancer_type').sample(20, random_state=42)
            umap_tmp = umap_data.loc[label_tmp.index]
        a = int(len(umap_tmp) * 0.25)
        b = int(len(umap_tmp) * 0.75)
        rejection_rates = bisection_compute_rejection_rates(
            umap_tmp, label_tmp['cancer_type'], a, b)
        kBET_col.append(rejection_rates)
        ARI_col.append(ce.adjusted_rand_index())
        MI_col.append(ce.mutual_info_score())
        silhouette_col.append(ce.silhouette_score())
        ps_col.append(ps)
        data_col.append(data_type)
    
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col,
                                 'silhouette_score':silhouette_col,
                                 'kBETscore':kBET_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/batch_evaluate/ps/' + data_type + '_cluster_score.tsv', sep='\t')


def multiprocessing_ps_evaluate():
    data_list = ['rpca_counts_pca', 'rpca_tpm_pca', 'cca_counts_pca', 'cca_tpm_pca',
                 'liger_counts_pca', 'liger_tpm_pca', 'harmony_counts_pca', 'harmony_tpm_pca',
                 'ruvr_counts', 'ruvs_counts',
                 'combat_counts', 'combat_tpm','combat_seq_counts',
                 'limma_counts', 'limma_tpm',
                 'zc', 'bfx', 'x']
    
    ps_list = ["Bladder", "Breast", "Colon", "Corpus uteri", "Kidney",
               "Liver and intrahepatic bile ducts", "Prostate gland",
               "Stomach", "Thyroid gland"]
    
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')

    import multiprocessing
    process_A = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[0], label, ps_list))
    process_B = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[1], label, ps_list))
    process_C = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[2], label, ps_list))
    process_D = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[3], label, ps_list))
    process_E = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[4], label, ps_list))
    process_F = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[5], label, ps_list))
    process_G = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[6], label, ps_list))
    process_H = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[7], label, ps_list))
    process_I = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[8], label, ps_list))
    process_J = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[9], label, ps_list))
    process_K = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[10], label, ps_list))
    process_L = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[11], label, ps_list))
    process_M = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[12], label, ps_list))
    process_N = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[13], label, ps_list))
    process_O = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[14], label, ps_list))
    process_P = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[15], label, ps_list))
    process_Q = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[16], label, ps_list))
    process_R = multiprocessing.Process(target=ps_evaluate,
                                        args=(data_list[17], label, ps_list))
    
    process_A.start()
    process_B.start()
    process_C.start()
    process_D.start()
    process_E.start()
    process_F.start()
    process_G.start()
    process_H.start()
    process_I.start()
    process_J.start()
    process_K.start()
    process_L.start()
    process_M.start()
    process_N.start()
    process_O.start()
    process_P.start()
    process_Q.start()
    process_R.start()


def subtype_batch_evaluate(data_type, label, ps_list):
   
    data_path = './data/output/benchmark_batch_correction'
    umap_path = os.path.join(data_path, (data_type + '_umap.tsv'))
    umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
    umap_data.index = label.index
    
    ps_col, data_col, ARI_col, MI_col, silhouette_col, kBET_col = [], [], [], [], [], []
    for ps in ps_list:
        label_tmp = label[(label['primary_site']==ps)& \
                              (label['cancer_type']!='Normal')]
        counts = label_tmp['cancer_type'].value_counts()
        drop = counts[counts <= 10].index.tolist()
        for i in drop:
            label_tmp[label_tmp==i] = np.nan
        label_tmp.dropna(how='any',inplace=True)
        umap_tmp = umap_data.loc[label_tmp.index]
        ce = clustering_evaluation(umap_tmp, label_tmp, label_class='cancer_type')
        if len(label_tmp)>40:
            label_tmp = label_tmp.groupby('cancer_type').sample(20, random_state=42)
            umap_tmp = umap_data.loc[label_tmp.index]
        a = int(len(umap_tmp) * 0.25)
        b = int(len(umap_tmp) * 0.75)
        rejection_rates = bisection_compute_rejection_rates(
            umap_tmp, label_tmp['cancer_type'], a, b)
        kBET_col.append(rejection_rates)
        ARI_col.append(ce.adjusted_rand_index())
        MI_col.append(ce.mutual_info_score())
        silhouette_col.append(ce.silhouette_score())
        ps_col.append(ps)
        data_col.append(data_type)
    
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col,
                                 'silhouette_score':silhouette_col,
                                 'kBETscore':kBET_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/batch_evaluate/subtype/' + data_type + '_cluster_score.tsv', sep='\t')


def multiprocessing_subtype_evaluate():
    data_list = ['rpca_counts_pca', 'rpca_tpm_pca', 'cca_counts_pca', 'cca_tpm_pca',
                 'liger_counts_pca', 'liger_tpm_pca', 'harmony_counts_pca', 'harmony_tpm_pca',
                 'ruvr_counts', 'ruvs_counts',
                 'combat_counts', 'combat_tpm','combat_seq_counts',
                 'limma_counts', 'limma_tpm',
                 'zc', 'bfx', 'x']
    
    ps_list = ["Adrenal gland", "Bronchus and lung", "Kidney"]
    
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')

    import multiprocessing
    process_A = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[0], label, ps_list))
    process_B = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[1], label, ps_list))
    process_C = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[2], label, ps_list))
    process_D = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[3], label, ps_list))
    process_E = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[4], label, ps_list))
    process_F = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[5], label, ps_list))
    process_G = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[6], label, ps_list))
    process_H = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[7], label, ps_list))
    process_I = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[8], label, ps_list))
    process_J = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[9], label, ps_list))
    process_K = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[10], label, ps_list))
    process_L = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[11], label, ps_list))
    process_M = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[12], label, ps_list))
    process_N = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[13], label, ps_list))
    process_O = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[14], label, ps_list))
    process_P = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[15], label, ps_list))
    process_Q = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[16], label, ps_list))
    process_R = multiprocessing.Process(target=subtype_batch_evaluate,
                                        args=(data_list[17], label, ps_list))
    
    process_A.start()
    process_B.start()
    process_C.start()
    process_D.start()
    process_E.start()
    process_F.start()
    process_G.start()
    process_H.start()
    process_I.start()
    process_J.start()
    process_K.start()
    process_L.start()
    process_M.start()
    process_N.start()
    process_O.start()
    process_P.start()
    process_Q.start()
    process_R.start()

if __name__ == '__main__':

    # multiprocessing_normal_evaluate()
    # multiprocessing_ps_evaluate()
    multiprocessing_subtype_evaluate()


        
    
    
    

 
