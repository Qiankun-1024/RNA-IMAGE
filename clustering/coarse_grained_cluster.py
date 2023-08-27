import time
import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn import cluster
from itertools import cycle, islice
from sklearn.metrics.cluster import homogeneity_score

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import log2_feature, normalize_feature


def generate_attrs(cancer_type, attr_threshold, freq_threshold):
    attrs = pd.read_csv('./data/output/attrs/all/sampling_diff/{}/{}_diff_attrs.tsv'.\
                        format(cancer_type, cancer_type),
                        sep='\t')
    res = pd.read_csv('./data/output/attrs/all/sampling_diff/{}/{}_diff_res.tsv'.\
                      format(cancer_type, cancer_type),
                      sep='\t', index_col=0)
    res['freq'] = res['freq']/len(attrs)
    res = res[res['freq'] > 0.05]
    res = res[(res['important_attrs'] > attr_threshold) | (res['freq'] > freq_threshold)]
    res = res.sort_values(by='important_attrs', axis=0, ascending=False)
    important_gene = res['gene']
    return important_gene


def generate_data(data, dtype, label, gene_id): 
    data = data.loc[label.index, gene_id]
    # data = data.loc[:,(data==0).count()/len(data)<=0.9]
    print(data)
    if dtype == 'TPM':
        data = log2_feature(data, dtype=dtype)
        data = normalize_feature(data)
    elif dtype == 'log2TPM':
        data = normalize_feature(data)
    else:
        data = data
    return data


def generate_label(cancer_type):
    meta = pd.read_csv('./data/TCGA_GTEx/meta_info.tsv',
                sep='\t')
    subtype = pd.read_csv('./data/TCGA_GTEx/tcga_subtype_label.tsv',
                          sep='\t', index_col=0)
    clinical = pd.read_csv('./data/TCGA_GTEx/clinical.tsv',
                    sep='\t', index_col=0)
    
    label = meta[meta['cancer_type']==cancer_type]
    label.index = label['File ID']
    label = pd.merge(label, subtype[['subtype']], how='left',
                     left_index=True, right_index=True)
    label = pd.merge(label, clinical, how='left',
                     left_index=True, right_index=True)
    return label

raw_data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv',
                       sep='\t', index_col=0)
datasets = [
    ('raw_data', raw_data, 'log2TPM')
    ]
thresholds = [0.2]

label = generate_label('BRCA')
print(label)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_label = le.fit_transform(label['subtype'].astype('str'))

plt.figure(figsize=(12, 20))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)
plot_num = 1
i_dataset = 1
pred_label, data_algorithm = [], []
for (data_name, dataset, dtype) in datasets:
    for t in thresholds:
        if dtype == 'latent_z':
            data = dataset.loc[label.index]
        else:
            important_gene = generate_attrs('BRCA', t, 0.5)
            data = generate_data(dataset, dtype, label, important_gene)
        
        from umap import UMAP
        X = UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(data)     
        
        emdedding = X
        n_clusters = 5
        
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=10)

        clustering_algorithms = [
            ('KMeans', kmeans)
        ]
        
        for i_algorithm, (name, algorithm) in enumerate(clustering_algorithms):
            t0 = time.time()
            algorithm.fit(emdedding)
            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(emdedding)
                
                
            PAM50_Homogeneity = homogeneity_score(
                label['subtype'][~label['subtype'].isna()],
                y_pred[~label['subtype'].isna()]
                )
            
            plt.subplot(2*len(thresholds)+1, len(clustering_algorithms)+1, plot_num)
            if i_dataset == 1:
                plt.title(name, size=18)
            if i_algorithm == 0:
                plt.ylabel('{}\n()d={}'.format(data_name, data.shape[1]))

            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y_pred) + 1),
                    )
                )
            )

            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], c=y_pred,
                        s=0.5, cmap='Spectral')

            plt.xticks(())
            plt.yticks(())
            plt.text(
                0.99,
                0.01,
                ("%.2fs" % (t1 - t0)).lstrip("0"),
                transform=plt.gca().transAxes,
                size=15,
                horizontalalignment="right",
            )
            plt.text(
                0.01,
                0.1,
                'PAM50 Homogeneity:{:.2f}'.format(PAM50_Homogeneity),
                transform=plt.gca().transAxes,
                size=10,
                horizontalalignment="left",
            )

            plot_num += 1
            
            pred_label.append(y_pred)
            data_algorithm.append('{}_{}_{}'.format(data_name, t, name.strip('\n')))
        
        plt.subplot(2*len(thresholds)+1, len(clustering_algorithms)+1, plot_num)
        plt.scatter(X[:, 0], X[:, 1], c=encoded_label,
                    s=0.5, cmap='Spectral')
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
        
        i_dataset += 1
        if dtype == 'latent_z':
            break

plt.savefig('./figure/clustering/cluster_algorithms.png')

pred_label = pd.DataFrame(pred_label,
                          index = data_algorithm,
                          columns = label.index
                          ).T
pred_label.to_csv('./data/output/clustering/coare_grained/cluster_label.tsv',
                  sep='\t')
