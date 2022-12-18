import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from evaluate import scaled_pc_regression

def umap_dimention_reduction(data):
    pca = PCA(n_components=100)
    pc = pca.fit_transform(data)
    X = UMAP(n_neighbors=25,
             min_dist=0.5).fit_transform(pc)
    X = pd.DataFrame(X)
    X.columns = ['UMAP1', 'UMAP2']
    return X


class clustering_evaluation():
    def __init__(self, umap_data, label, label_class='primary_site'):
        self.data = umap_data
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
        
    label = pd.read_csv('./data/all_label.tsv', sep='\t', index_col=0)
    path = '/vol1/cuipeng_group/qiankun/GAN-VAE/figure/TSNE'
    data_list = ['seurat_counts', 'seurat_tpm', 'combat_tpm', 'ruvr_counts',
                 'ruvs_counts', 'limma_counts', 'combat_counts',
                 'x', 'z', 'bf_x']
    ps_list = ["Bladder", "Colon", "Corpus uteri", "Kidney",
               "Liver and intrahepatic bile ducts", "Prostate gland",
               "Stomach", "Thyroid gland"]
    avg_matrix = pd.DataFrame()    
    ps_col, data_col, ARI_col, MI_col, silhouette_col= [], [], [], [], [] 
    for i in data_list:
        umap_path = os.path.join(path, (i + '_umap.tsv'))
        if os.path.exists(umap_path):
            umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
        else:
            x = pd.read_csv(os.path.join('./data/output/', (i + '.tsv')),
                            sep='\t', index_col=0)
            if i not in ['seurat_counts', 'seurat_tpm', 'z']:
                x[x<0] = 0
                x = np.log2(x + 1)
            umap_data = umap_dimention_reduction(x)
            umap_data.to_csv(umap_path, sep='\t')
        umap_data.index = label.index
        ARI_score, MI_score, silhouette_score = 0, 0, 0
        for ps in ps_list:
            label_tmp = label[(label['primary_site']==ps)& \
                              (label['cancer_type']=='Normal')]
            umap_tmp = umap_data.loc[label_tmp.index]
            ce = clustering_evaluation(umap_tmp, label_tmp, label_class='batch')
            ARI_score += ce.adjusted_rand_index()
            MI_score += ce.mutual_info_score()
            silhouette_score += ce.silhouette_score()
            ARI_col.append(ce.adjusted_rand_index())
            MI_col.append(ce.mutual_info_score())
            silhouette_col.append(ce.silhouette_score())
            ps_col.append(ps)
            data_col.append(i)
        ARI_score /= len(ps_list)
        MI_score /= len(ps_list)
        silhouette_score /= len(ps_list)
        avg_matrix[i] = [ARI_score, MI_score, silhouette_score]
    
    pcr_list = []
    pcr_col = []
    pcr = scaled_pc_regression()
    data_list = ['seurat_counts_pca', 'seurat_tpm_pca', 'combat_tpm',
                 'ruvr_counts', 'ruvs_counts', 'limma_counts',
                 'combat_counts', 'x_tpm', 'z', 'bfx_tpm']
    for i in data_list:
        x = pd.read_csv(os.path.join('./data/output/', (i + '.tsv')),
                            sep='\t', index_col=0)
        if i not in ['seurat_counts', 'seurat_tpm', 'z']:
            x[x<0] = 0
            x = np.log2(x + 1)
        pcr_score = 0
        for ps in ps_list:
            label_tmp = label[(label['primary_site']==ps)& \
                              (label['cancer_type']=='Normal')]
            x_tmp = x.loc[label_tmp.index]
            _, score = pcr(x_tmp, label_tmp['batch'], 10)
            pcr_score += score
            pcr_col.append(score)
        pcr_score /= len(ps_list)
        pcr_list.append(pcr_score)
    avg_matrix.loc['pcr_score'] = pcr_list
    avg_matrix.to_csv('./data/output/ps_cluster_score.tsv', sep='\t')
    
    score_matrix = pd.DataFrame({'ARI_score':ARI_col,
                                 'MI_score':MI_col, 
                                 'silhouette_score':silhouette_col,
                                 'pcr_score':pcr_col,
                                 'primary_site':ps_col,
                                 'method':data_col})
    score_matrix.to_csv('./data/output/split_cluster_score.tsv', sep='\t')
    
    
    # breast score
    data_list = ['x', 'z', 'bfx']
    ps_list = ["Breast"]
    label = pd.read_csv('./data/gtex_breast/label.tsv', sep='\t', index_col=0)
    for i in data_list:
        umap_path = './data/gtex_breast/' + i + '_umap.tsv'
        umap_data = pd.read_csv(umap_path, sep='\t', index_col=0)
        umap_data.index = label.index
        ARI_score, MI_score, silhouette_score = 0, 0, 0
        
        label_tmp = label[(label['primary_site']=="Breast")& \
                            (label['cancer_type']=='Normal')]
        umap_tmp = umap_data.loc[label_tmp.index]
        ce = clustering_evaluation(umap_tmp, label_tmp, label_class='batch')
        ARI_score = ce.adjusted_rand_index()
        MI_score = ce.mutual_info_score()
        silhouette_score = ce.silhouette_score()
        print(ARI_score, MI_score, silhouette_score)
        
    data_list = ['x', 'z', 'bf_x']
    for i in data_list:
        x = pd.read_csv(os.path.join('./data/gtex_breast/', (i + '.tsv')),
                            sep='\t', index_col=0)
        if i not in ['seurat_counts', 'seurat_tpm', 'z']:
            x[x<0] = 0
            x = np.log2(x + 1)
        pcr_score = 0

        label_tmp = label[(label['primary_site']=="Breast")& \
                            (label['cancer_type']=='Normal')]
        x_tmp = x.loc[label_tmp.index]
        _, score = pcr(x_tmp, label_tmp['batch'], 10)
        pcr_score = score
        print(pcr_score)
    
    
        

    