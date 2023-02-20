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
        


if __name__ == '__main__':
    '''
    switch GPU
    '''
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")
        print(gpu)
    
    # normal tissue integrate evaluation
    data_path = './data/output/benchmark_batch_correction'
    data_list = ['rpca_counts_pca', 'rpca_tpm_pca', 'cca_counts_pca', 'cca_tpm_pca',
                 'liger_counts_pca', 'liger_tpm_pca', 'harmony_counts_pca', 'harmony_tpm_pca',
                 'ruvr_counts', 'ruvs_counts',
                 'combat_counts', 'combat_tpm','combat_seq_counts',
                 'limma_counts', 'limma_tpm',
                 'zc', 'bfx', 'x']
    
    ps_list = ["Bladder", "Colon", "Corpus uteri", "Kidney",
               "Liver and intrahepatic bile ducts", "Prostate gland",
               "Stomach", "Thyroid gland"]
    
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                        index_col=0, sep='\t')
    ps_col, data_col, ARI_col, MI_col, silhouette_col, kBET_col = [], [], [], [], [], []
    for i in data_list:
        umap_path = os.path.join(data_path, (i + '_umap.tsv'))
        umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
        umap_data.index = label.index
        for ps in ps_list:
            label_tmp = label[(label['primary_site']==ps)& \
                              (label['cancer_type']=='Normal')]
            umap_tmp = umap_data.loc[label_tmp.index]
            ce = clustering_evaluation(umap_tmp, label_tmp, label_class='batch')
            a = int(len(umap_tmp) * 0.25)
            b = int(len(umap_tmp) * 0.75)
            rejection_rates = bisection_compute_rejection_rates(
                umap_tmp, label_tmp['batch'], a, b)
            kBET_col.append(rejection_rates)
            ARI_col.append(ce.adjusted_rand_index())
            MI_col.append(ce.mutual_info_score())
            silhouette_col.append(ce.silhouette_score())
            ps_col.append(ps)
            data_col.append(i)
        
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col,
                                 'silhouette_score':silhouette_col,
                                 'kBETscore':kBET_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/normal_sample_cluster_score.tsv', sep='\t')
    
    # subtype differentiation evaluation
    ps_list = ["Adrenal gland", "Bronchus and lung", "Kidney"]
    ps_col, data_col, ARI_col, MI_col, silhouette_col, kBET_col = [], [], [], [], [], []
    for i in data_list:
        umap_path = os.path.join(data_path, (i + '_umap.tsv'))
        umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
        umap_data.index = label.index
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
            a = int(len(umap_tmp) * 0.25)
            b = int(len(umap_tmp) * 0.75)
            rejection_rates = bisection_compute_rejection_rates(
                umap_tmp, label_tmp['cancer_type'], a, b)
            kBET_col.append(rejection_rates)
            ARI_col.append(ce.adjusted_rand_index())
            MI_col.append(ce.mutual_info_score())
            silhouette_col.append(ce.silhouette_score())
            ps_col.append(ps)
            data_col.append(i)
        
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col,
                                 'silhouette_score':silhouette_col,
                                 'kBETscore':kBET_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/subtype_cluster_score.tsv', sep='\t')
    


        
    
    
    

 
