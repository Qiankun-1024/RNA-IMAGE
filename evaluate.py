import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table 
from scipy.stats import chisquare

from data_preprocess import *


class data_visualization():
    '''visualizes the batch effect correction of a class of data'''

    @staticmethod
    def data_filter(data, label, selected_classes):
        for k, v in selected_classes.items():
            label = label[label[k].isin(v)]
        data = data.loc[label.index]
        return data, label
    
    @staticmethod
    def pca(data, label, plt_classes):
        le = LabelEncoder()
        pca = PCA(n_components=2,
                  random_state=42)
        embedding = pca.fit_transform(data)
        explained_variance_ratio_ = pca.explained_variance_ratio_

        fig = plt.figure(figsize=(8 * len(plt_classes), 8))
        for i, c in enumerate(plt_classes):
            encoded_label = le.fit_transform(label[c].astype('str'))
            ax = plt.subplot(1, len(plt_classes), i+1)
            scatter = ax.scatter(embedding[:,0], embedding[:,1],
                                 alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
            plt.xlabel('PC1({:.2%} explained variance)'.\
                format(explained_variance_ratio_[0]))
            plt.ylabel('PC2({:.2%} explained variance)'.\
                format(explained_variance_ratio_[1]))
            plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                    labels=list(np.unique(label[c].astype('str'))), scatterpoints=1)
        embedding = pd.DataFrame(embedding)
        embedding.columns = ['PC1', 'PC2']   
        return fig, embedding
    
    @staticmethod   
    def tsne(data, label, plt_classes, perplexity=30):
        le = LabelEncoder()
        embedding = TSNE(
            n_components=2,
            init='pca', 
            random_state=42, 
            perplexity=perplexity
            ).fit_transform(data)

        fig = plt.figure(figsize=(8 * len(plt_classes), 8))
        for i, c in enumerate(plt_classes):
            encoded_label = le.fit_transform(label[c].astype('str'))
            ax = plt.subplot(1, len(plt_classes), i+1)
            scatter = ax.scatter(embedding[:,0], embedding[:,1],
                                 alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
            plt.xlabel('tSNE-1')
            plt.ylabel('tSNE-2')
            plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                    labels=list(np.unique(label[c].astype('str'))), scatterpoints=1)
        embedding = pd.DataFrame(embedding)
        embedding.columns = ['tSNE-1', 'tSNE-2']        
        return fig, embedding
    
    @staticmethod    
    def umap_3D(data, label, plt_classes, n_neighbors=15, min_dist=0.1):
        le = LabelEncoder()
        embedding = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
            ).fit_transform(data)

        fig = plt.figure(figsize=(8 * len(plt_classes), 8))
        for i, c in enumerate(plt_classes):
            encoded_label = le.fit_transform(label[c].astype('str'))
            ax = plt.subplot(1, len(plt_classes), i+1, projection='3d')
            scatter = ax.scatter(embedding[:,0],
                                 embedding[:,1],
                                 embedding[:,2],
                                 alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
            plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                    labels=list(np.unique(label[c].astype('str'))), scatterpoints=1)
        embedding = pd.DataFrame(embedding)
        embedding.columns = ['UMAP1', 'UMAP2', 'UMAP3']
        return fig, embedding
        
    @staticmethod
    def umap(data, label, plt_classes, n_neighbors=15, min_dist=0.1):
        
        le = LabelEncoder()
        embedding = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
            ).fit_transform(data)

        fig = plt.figure(figsize=(8 * len(plt_classes), 8))
        for i, c in enumerate(plt_classes):
            encoded_label = le.fit_transform(label[c].astype('str'))
            ax = plt.subplot(1, len(plt_classes), i+1)
            scatter = ax.scatter(embedding[:,0], embedding[:,1],
                                 alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                    labels=list(np.unique(label[c].astype('str'))), scatterpoints=1)
        
        embedding = pd.DataFrame(embedding)
        embedding.columns = ['UMAP1', 'UMAP2']
        return fig, embedding
    
        

class scaled_pc_regression():
    '''
    pc regression to detect linear batch effects
    k:top k pc
    '''
    def __call__(self, data, batch, k):
        self.data = data
        le = LabelEncoder()
        self.batch = le.fit_transform(batch)
        self.k = k
        self.pc, variance = self.pca()
        coef, pvalues, rsquared = self.ols()
        pc_score = self.pc_score(pvalues, variance)
        summary_df = pd.DataFrame({'coef':coef, 'pvalues':pvalues, 'rsquared':rsquared, 'pc_score':pc_score})
        return summary_df, pc_score


    def pca(self):
        pca = PCA(n_components=self.k)
        pc = pca.fit_transform(self.data)
        explained_variance_ratio_ = pca.explained_variance_ratio_
        return pc, explained_variance_ratio_

    def ols(self):
        coef, pvalues, rsquared = [], [], []
        for i in range(self.k):
            model = sm.OLS(self.batch,self.pc[:,i:i+1]).fit()
            coef.append(float(model.params))
            pvalues.append(float(model.pvalues))
            rsquared.append(float(model.rsquared))
        return np.array(coef), np.array(pvalues), np.array(rsquared)
    
    def pc_score(self, pvalues, variance):
        pindex = np.where(pvalues < 0.05, 0, 1)
        p_score = np.sum(np.multiply(variance, pindex))/np.sum(variance)
        return p_score



class k_neighbours_chisquare_test():
    '''
    k nearest neighbours rejection rate
    '''
    def __call__(self, data, batch, k):
        self.batch = batch
        pca = PCA(n_components=10)
        pc = pca.fit_transform(data)
        
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
        


def tsne_visualization(data, label, out_name):
    pca = PCA(n_components=100)
    data = pca.fit_transform(data)
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)
    X = TSNE(n_components=2, init='pca',
                            random_state=0, perplexity=100).fit_transform(data)

    fig = plt.figure(figsize=(15, 8))
    scatter = plt.scatter(X[:,0], X[:,1], s=1, alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.legend(handles = scatter.legend_elements()[0], bbox_to_anchor=(1.05, 1),
                 loc=2, borderaxespad=0, shadow=False, labels=list(np.unique(label)), scatterpoints=1)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    fig.subplots_adjust(right=0.7)

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure', 'TSNE')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (out_name+'.png')))
    plt.close()
    X_df = pd.DataFrame(X)
    X_df.columns = ['TSNE-1', 'TSNE-2']
    X_df.to_csv(os.path.join(plt_dir, (out_name+'.tsv')), sep='\t')



def load_meta_label_data(data_path, primary_site, label_type):
    # x = pd.read_csv(os.path.join(data_path, 'output', 'x_tpm.tsv'), sep='\t', index_col=0)
    z = pd.read_csv(os.path.join(data_path, 'output', 'z.tsv'), sep='\t', index_col=0)
    # bf_x = pd.read_csv(os.path.join(data_path, 'output', 'bfx_tpm.tsv'), sep='\t', index_col=0)
    # combat = pd.read_csv(os.path.join(data_path, 'output', 'combat_x.tsv'), sep='\t', index_col=0).T
    label = pd.read_csv('/vol1/cuipeng_group/qiankun/CVAE/data/labels.tsv', sep='\t', index_col=0)
    # x = log2_feature(x, data_type='TPM')
    # x = log2_feature(bf_x, data_type='TPM')

    selected_label = label[(label['primary_site']==primary_site) & (label[label_type]!='None') & (label['cancer_type']=='KIRC')]
    sample = list(set(list(selected_label.index)).intersection(set(list(z.index))))
    selected_label = selected_label.loc[sample]
    # selected_x = x.loc[sample]
    selected_z = z.loc[sample]
    # selected_bfx = bf_x.loc[sample]
    # selected_combat = combat.loc[sample]
    # print(selected_combat)

    return selected_z, selected_label[label_type]
    # return selected_x, selected_z, selected_bfx, selected_combat, selected_label



def pca(data, label, out_name):
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(data)
    explained_variance_ratio_ = pca.explained_variance_ratio_

    scatter = plt.scatter(pc[:,0], pc[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
    plt.xlabel('PC1({:.2%} explained variance)'.format(explained_variance_ratio_[0]))
    plt.ylabel('PC2({:.2%} explained variance)'.format(explained_variance_ratio_[1]))
    plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                labels=list(np.unique(label)), scatterpoints=1)
    

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (out_name+ '.png')))



def matrix_heatmap():
    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    z_matrix = pd.DataFrame({'zc':[0.55,0.96],'zb':[1.0,0.46]})
    z_matrix.index = ['batch', 'category']
    ax = sns.heatmap(z_matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    annot=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(plt_dir, 'z_heatmap.png'))
    plt.close()

    x_matrix = pd.DataFrame({'x':[0.99,0.99],'bf_x':[0.51,0.88]})
    x_matrix.index = ['batch', 'category']
    ax = sns.heatmap(x_matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    annot=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(plt_dir, 'x_heatmap.png'))
    plt.close()






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

    #######################################################################################
    # out_path = os.path.join(data_path, 'output', 'pcr')
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # # data_list = ['combat_log2tpm', 'limma_log2tpm' ,'combat_tpm', 'limma_tpm']
    # # data_list = ['ruvs_counts', 'limma_log2counts', 'limma_counts', 'combat_counts']
    # data_list = ['x_tpm']
    # pcr = scaled_pc_regression()
    # label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/y_data.tsv', sep='\t', index_col=0)
    # for i in data_list:
    #     data = pd.read_csv(os.path.join(data_path, 'output', (i + '.tsv')), sep='\t', index_col=0)
    #     output = pcr(data, label['primary_site'], 20)
    #     output.to_csv(os.path.join(out_path, (i + '_pcr.tsv')), sep='\t')
    #######################################################################################
    # data_list = ['limma_tpm']
    # for i in data_list:
    #     data = pd.read_csv(os.path.join(data_path, 'output', (i+'.tsv')), sep='\t', index_col=0)
    #     tsne_visualization(data, label['primary_site'], (i+'_tsne'))
    #######################################################################################
    # data = pd.read_csv(os.path.join(data_path, 'output', 'cpu1_ckpt56_z.tsv'), sep='\t', index_col=0)
    # label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/y_data.tsv', sep='\t', index_col=0)
    # rejection_rates = []
    # for name, group in label.groupby(['cancer_type', 'primary_site']):
    #     if group['batch'].nunique() > 1:
    #         group_data = data.loc[group.index]
    #         a = int(len(group_data) * 0.25)
    #         b = int(len(group_data) * 0.75)
    #         score = bisection_compute_rejection_rates(group_data, group['batch'], a, b)
    #         #score = knt(group_data, group['batch'], int(len(group_data)/2))
    #         rejection_rates.append(score)
    # rejection_rates = np.mean(rejection_rates)
    # print(rejection_rates)
    
    #######################################################################################
    # data_list = ['cpu1_ckpt56_bf_x', 'test_x', 'cpu1_ckpt56_z', 'combat_counts', 'combat', 'limma_log2counts',
    #              'limma', 'limma_counts', 'ruvr_counts', 'ruvs_counts']
    # # data_list = ['cpu1_ckpt56_bf_x', 'cpu1_ckpt56_z']
    # score = pd.DataFrame()
    # for i in data_list:
    #     label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/y_data.tsv', sep='\t', index_col=0)
    #     ce = cp(i, label)
    #     ari = ce.adjusted_rand_index()
    #     ami = ce.mutual_info_score()
    #     score[i] = [ari, ami]
    # print(score)

        
    
    
    

 
