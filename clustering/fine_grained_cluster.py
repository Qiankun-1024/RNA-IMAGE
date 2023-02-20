import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import log2_feature, normalize_feature

def BIC(data):
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_data = ro.conversion.py2rpy(data)
    
    importr('mclust')
    mod = ro.r['Mclust'](r_data)
    return int(mod[5]), mod[10]


def coarse_grained_attrs(cancer_type,threshold):
    attrs = pd.read_csv('./data/output/attrs/all/sampling_diff/{}/{}_diff_attrs.tsv'.\
                        format(cancer_type, cancer_type),
                        sep='\t')
    res = pd.read_csv('./data/output/attrs/all/sampling_diff/{}/{}_diff_res.tsv'.\
                      format(cancer_type, cancer_type),
                      sep='\t', index_col=0)
    res['freq'] = res['freq']/len(attrs)
    res = res[res['freq'] > 0.05]
    res = res.sort_values(by='important_attrs', axis=0, ascending=False)
    important_gene = res['gene'][0:threshold]
    return important_gene


def generate_attrs(clus_id,threshold):
    attrs = pd.read_csv('./data/output/attrs/Breast/subtype/cluster_{}_attrs.tsv'.\
                        format(clus_id),
                        sep='\t')
    res = pd.read_csv('./data/output/attrs/Breast/subtype/cluster_{}_res.tsv'.\
                      format(clus_id),
                      sep='\t', index_col=0)
    res['freq'] = res['freq']/len(attrs)
    res = res[res['freq'] > 0.05]
    res = res[res['important_attrs'] > threshold]
    res = res.sort_values(by='important_attrs', axis=0, ascending=False)
    important_gene = res['gene']
    return important_gene


def mean_euclidean_distance(data):
    ab = np.dot(data, data.T)
    a2 = np.matmul(data**2, np.ones(data.shape).T)
    b2 = a2.T
    dist_mat = a2 + b2 - (2 * ab)
    mean_dist = np.mean(dist_mat)
    return mean_dist
    

cluster_label = pd.read_csv('./attr_src/cluster_label.tsv',
                            index_col=0, sep='\t')
meta = pd.read_csv('./data/meta_info.tsv',
                sep= '\t', index_col=0)
cluster_label.index = meta.loc[cluster_label.index, 'entity_submitter_id'] 
cluster_label['cluster'] = cluster_label['raw_data_0.2_KMeans']
data = pd.read_csv('./data/Breast/breast_data.tsv',
                    index_col=0, sep='\t')
data = log2_feature(data, dtype='TPM')
data = normalize_feature(data)

plt.figure(figsize=(12, 20))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)
plot_num = 1
nrow = 1
thresholds = [0.2]
clusters = [9, 6, 7, 8, 9]
silhouette_score = pd.DataFrame()
mean_distance = pd.DataFrame()
bic_score = pd.DataFrame()

for clus_id in np.unique(cluster_label['cluster']):
    pred_label = []
    tmp_label = cluster_label[cluster_label['cluster']==clus_id]
    tmp_data = data.loc[tmp_label.index]
    silhouette_result = []
    dist_result = []
    bic_result = []

    for i, t in enumerate(thresholds):
        ig = generate_attrs(clus_id, t)
        tmp = tmp_data[ig]
        
        X = UMAP(
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=42,
        ).fit_transform(tmp)     
        
        embedding = X
        
        n_clusters = clusters[clus_id]
        
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embedding)
        y_pred = kmeans.labels_.astype(int)
        
        silhouette_result.append(metrics.silhouette_score(
            embedding, kmeans.labels_, metric='euclidean'
            ))
        dist_result.append(mean_euclidean_distance(embedding))
        
        plt.subplot(len(np.unique(cluster_label['cluster'])), len(thresholds), plot_num)
        if nrow == 1:
            plt.title(t, size=18)
        if i == 0:
            plt.ylabel('cluster {}'.format(clus_id), size=18)
            
        plt.scatter(X[:, 0], X[:, 1], c=y_pred,
                        s=0.5, cmap='Spectral')

        plt.xticks(())
        plt.yticks(())

        pred_label.append(y_pred)
        plot_num += 1 
    nrow += 1
        
    pred_label = pd.DataFrame(pred_label,
                        index = thresholds,
                        columns = tmp_label.index
                        ).T
    pred_label.to_csv('./data/output/clustering/fine_grained/cluster_{}.tsv'.format(clus_id),
                        sep='\t')
    
plt.savefig('./figure/clustering/fine_grained_cluster.png')

